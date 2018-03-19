import Utils.project
import Utils.random
import com.google.common.collect.Sets

fun main(args : Array<String>){
    val trueMDP = simpleBoutillier()

    val (_, trueValues) = structuredValueIteration(trueMDP.rewardTree, trueMDP.dbns)
    val trueQs = trueMDP.dbns.mapValues { (_, dbn) -> regress(trueValues, trueMDP.rewardTree, dbn)}
    val expert = Expert(10, 0.25, 10, HashSet(), trueMDP, trueQs)

    val aliveThresh = 0.01
    val singleParentProb = 0.1
    val pseudoCountSize = 20.0
    val buntineUpdater = BuntineUpdater(aliveThresh, pseudoCountSize, singleParentProb)

    var agentVocab = trueMDP.vocab.toSet()
    val agentActions = mutableSetOf("A1")
    val agentTrialHist = agentActions.associate { Pair(it, ArrayList<Pair<TimeStamp, SequentialTrial>>()) }.toMutableMap()
    val expertEv = emptyList<DirEdge>()
    val actionAdvice = HashMap<RVAssignment, Pair<TimeStamp, Action>>()

    val agentDBNInfos = agentActions.associate{ Pair(it, buntineUpdater.initialDBNInfo(agentVocab, expertEv, 0))}.toMutableMap()


    val initARewardScope = setOf(trueMDP.rewardScope.random())
    var agentRewardITI : RewardTree = RLeaf(initARewardScope, initARewardScope, createStats(initARewardScope, emptyList()), emptyList())
    var agentRewardDT = convertFromITI(agentRewardITI)
    var agentValueTree = agentRewardDT
    var agentPolicy = choosePolicy(agentDBNInfos.mapValues { (_, dbnInfo) -> regress(agentValueTree, agentRewardDT, dbnInfo.dbn) })
    val existRewardStates : MutableList<Trial> = ArrayList()

    var previousState = generateSample(trueMDP.vocab.toList())

    var tStep = 0
    while(tStep < 3000){
        // e greedy choice strategy at the moment
        val projectedPrevState = project(previousState, agentVocab)
        val chosenAction = eGreedy(agentActions, agentPolicy, previousState, 0.1)

        // Remember to project this onto agent's known vocabulary
        val newTrueState = generateSample(trueMDP.dbns[chosenAction]!!, previousState)
        val projectedNewState = project(newTrueState, agentVocab)
        val reward = matchLeaf(trueMDP.rewardTree, newTrueState).value

        val agentSeqTrial = Pair(tStep, SequentialTrial(projectedPrevState, chosenAction, projectedNewState, reward))
        agentTrialHist[chosenAction]!!.add(agentSeqTrial)
        expert.stateHist[tStep] = newTrueState

        // Reward Function Update
        val oldVocab = agentVocab
        val rewardTrial = Trial(newTrueState, reward)
        if(rewardTrial !in existRewardStates){
            val redundantTrial = existRewardStates.find { it.reward == rewardTrial.reward && partialMatch(it.assignment, rewardTrial.assignment) }
            if(redundantTrial != null){
                //If we havent seen this exact one before, but there is a trial that matches on a subset, *remove* it from the list of existential states
                existRewardStates.remove(redundantTrial)
                agentRewardITI = removeExample(agentRewardITI, redundantTrial)
            }

            existRewardStates.add(rewardTrial)

            // Consistency Check
            val solvableReward = unfactoredReward(agentRewardITI.vocab.toList() , existRewardStates)
            if(solvableReward == null){
                val newRewardVars = whatsInRewardScope(agentRewardITI.vocab, expert)
                tStep += 1
                agentRewardITI = addAdditionalVocab(agentRewardITI, newRewardVars)
                agentVocab += newRewardVars
            }

            agentRewardITI = incrementalUpdate(agentRewardITI, Trial(projectedNewState, reward))
            agentRewardDT = convertFromITI(agentRewardITI)
        }

        if(oldVocab != agentVocab){
            agentDBNInfos.mapValues { (a, dbnInfo) -> buntineUpdater.addBeliefVariables(agentVocab - oldVocab, tStep, agentTrialHist[a]!!.map{it.second}, expertEv, dbnInfo) }
            agentTrialHist.forEach{ _, trials -> trials.clear() }
            actionAdvice.clear()
        }
        else{
            agentDBNInfos[chosenAction] = buntineUpdater.trialUpdate(agentSeqTrial.second, agentTrialHist[chosenAction]!!.map{it.second}, expertEv, tStep, agentDBNInfos[chosenAction]!!)
        }


        // Better Action Advice
        expert.agentOptimalActionHistory.add(wasActionOptimal(previousState, chosenAction, trueQs))
        if(tStep - expert.lastAdvice < expert.adviceInterval && !expert.agentOptimalActionHistory.last() && poorRecentPerformance(expert.agentOptimalActionHistory, expert.historyWindow, expert.mistakeProportion)){
            // Then tell the agent about the better action
            val betterAction = betterActionAdvice(previousState, expert)
            tStep += 1

            if(betterAction !in agentActions){
                agentActions.add(betterAction)
                agentDBNInfos[betterAction] = buntineUpdater.initialDBNInfo(agentVocab, expertEv, tStep)
                agentTrialHist[betterAction] = ArrayList()
            }

            val (newValue, _) = applyExpertAdvice(agentRewardDT, agentValueTree, agentDBNInfos.mapValues{ it.value.dbn }, Pair(projectedPrevState, betterAction))
            agentValueTree = newValue


            // Resolving misunderstandings
            if(!actionAdvice.containsKey(projectedPrevState) || actionAdvice[projectedPrevState]!!.second == betterAction){
                actionAdvice[projectedPrevState] = Pair(agentSeqTrial.first, betterAction)
            }
            else{
                val earlierAdvice = actionAdvice[projectedPrevState]!!
                val latestAdvice = Pair(agentSeqTrial.first, betterAction)
                val resolution = resolveMisunderstanding(earlierAdvice.first, latestAdvice.first, expert)

                tStep += 1

                val resolutionVar = resolution.firstAssignment.first
                if(resolutionVar in agentVocab){
                    throw IllegalStateException("If differing variable was already in agent vocab, then why was there a misunderstanding?")
                }
                actionAdvice.clear()
                actionAdvice[projectedPrevState + resolution.firstAssignment] = earlierAdvice
                actionAdvice[projectedPrevState + resolution.secondAssignment] = latestAdvice

                agentVocab += resolutionVar
                agentDBNInfos.mapValues { (a, dbnInfo) -> buntineUpdater.addBeliefVariables(setOf(resolutionVar), tStep, agentTrialHist[a]!!.map{it.second}, expertEv, dbnInfo) }
                agentTrialHist.forEach { _, trials -> trials.clear() }
            }
        }

        // Policy Updates
        val (newValue, newQTrees) = incrementalSVI(agentRewardDT, agentValueTree, agentDBNInfos.mapValues {it.value.dbn} )
        agentValueTree = newValue
        agentPolicy = choosePolicy(newQTrees)

        previousState = newTrueState
    }
}


private fun cptsITIToDBNs(bestPInfos: Map<Action, Map<RandomVariable, SeqPInfo>>, cptsITI: Map<Action, Map<RandomVariable, ProbTreeITI>>, pseudoCountSize: Double): MutableMap<String, DynamicBayesNet> {
    return cptsITI
        .mapValues { (a, cpts) -> cptsITIToDBN(bestPInfos[a]!!, cpts, pseudoCountSize) }
        .toMutableMap()
}

fun cptsITIToDBN(bestPInfos : Map<RandomVariable, SeqPInfo>, cptsITI: Map<RandomVariable, ProbTreeITI>, pseudoCountSize: Double) =
    cptsITI.mapValues { (rv, dt) -> convertToCPT(dt, bestPInfos[rv]!!.priorParams, pseudoCountSize)}

private fun initialCPTsITI(agentActions: MutableSet<String>, agentVocab: Set<RandomVariable>,
                           bestPInfos: MutableMap<Action, Map<RandomVariable, SeqPInfo>>,
                           priorDBNs: MutableMap<String, DynamicBayesNet>, pseudoCountSize: Double): MutableMap<Action, Map<RandomVariable, ProbTreeITI>> =
    agentActions
        .associate { Pair(it, initialCPTsITI(agentVocab, bestPInfos[it]!!, priorDBNs[it]!!, pseudoCountSize)) }
        .toMutableMap()

fun initialCPTsITI(agentVocab: Set<RandomVariable>, bestPInfos: Map<RandomVariable, SeqPInfo>, priorDBN: DynamicBayesNet, pseudoCountSize: Double): Map<RandomVariable, ProbTreeITI> =
    agentVocab.associate { rv ->
        val bestPSet = bestPInfos[rv]!!.parentSet
        Pair(rv, ProbTreeITI(rv, bestPSet, priorDBN[rv]!!, pseudoCountSize))
    }

fun initialPSetPriors(actions : Set<Action>, beliefVocab : Set<RandomVariable>, singleParentProb : Double) =
    actions
        .associate { Pair(it, initialPSetPriors(beliefVocab, singleParentProb)) }
        .toMutableMap()

fun initialPSetPriors(beliefVocab: Set<RandomVariable>, singleParentProb: Double) : Map<RandomVariable, LogPrior<PSet>> =
    beliefVocab.associate { rv ->
        Pair(rv, minParentsPrior(beliefVocab, singleParentProb))
    }

data class ParentExtResult(val logPriors : Map<RandomVariable, LogPrior<PSet>>, val reasonableParents : Map<RandomVariable, List<SeqPInfo>>)

fun posteriorToPrior(reasonableParents : Map<RandomVariable, List<SeqPInfo>>, oldVocab : Set<RandomVariable>, extraVocab : Set<RandomVariable>, prevJointPriorParams: Map<RandomVariable, DecisionTree<Factor>>,
                     singleParentProb : Double, aliveThresh : Double, priorSampleSize : Double) : ParentExtResult {
    val inReasonable = Math.log(0.99)
    val outReasonable = Math.log(0.01)
    val totalPsets = numAssignments(oldVocab + extraVocab)

    val priorMaps = reasonableParents.mapValues { (_, rps) ->
        rps.flatMap { oldPInfo ->
            Sets.powerSet(extraVocab).map { extraSubset : Set<RandomVariable> ->
                val pSet = oldPInfo.parentSet + extraSubset
                val logProb = extraSubset.size * Math.log(singleParentProb) + (extraVocab.size - extraSubset.size) * Math.log(1 - singleParentProb) + oldPInfo.logProbability
                Pair(pSet, logProb)
            }
        }.associate{ it }
    }

    val priorFuncs : Map<RandomVariable, LogPrior<PSet>> = priorMaps.mapValues{ (_, priorMap) ->
        { pSet : PSet ->
            if(pSet in priorMap){
                inReasonable + priorMap[pSet]!!
            }
            else{
                outReasonable + 1.0 / (totalPsets - priorMap.size)
            }
        }
    }

    val bestPSetScores = priorMaps.mapValues { (_, rps) -> rps.values.max() }

    val newReasonable = priorMaps.mapValues { (rv, rps) ->
        rps.filterValues { it < bestPSetScores[rv]!! * aliveThresh }
            .map { (pSet, _) ->  createPInfo(rv, pSet, emptyList(), priorFuncs[rv]!!, prevJointPriorParams[rv]!!, priorSampleSize, emptyList())}
    }

    return ParentExtResult(priorFuncs, newReasonable)
}

fun eGreedy(actions : Set<Action>, exploitPolicy : PolicyTree, state : RVAssignment, exploreAmount : Double) : Action =
    if(Math.random() < 1 - exploreAmount) matchLeaf(exploitPolicy, state).value else actions.random()

