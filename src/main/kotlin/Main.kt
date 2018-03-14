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
    val priorJointParams = agentActions.associate { Pair(it, unifStartIDTransJointDBN(agentVocab, 0.1)) }.toMutableMap()

    val agentTrialHist = agentActions.associate { Pair(it, ArrayList<SequentialTrial>()) }.toMutableMap()
    val expertEv = emptyList<DirEdge>()
    val actionAdvice = HashMap<RVAssignment, Pair<TimeStamp, Action>>()

    val agentDBNInfos : Map<Action, DBNInfo> = buntineUpdater.initialDBNInfo(agentActions, agentVocab, expertEv)

    // 1. Set Buntine Prior
    val pSetPriors = initialPSetPriors(agentActions, agentVocab, singleParentProb)

    // 2. Set Initial Reasonable Parents and probabilities
    val reasonableParents = pSetPriors.mapValues { (action, priors) ->
        structuralUpdate(agentVocab, emptyList(), expertEv, priors, aliveThresh, priorJointParams[action]!!, pseudoCountSize)
    }.toMutableMap()

    val bestPInfos = reasonableParents.mapValues { (_, rps) -> bestParents(rps) }.toMutableMap()

    // 3. Set initial DBNs based on reasonable parents, zero counts, prior dbn...
    val cptsITI = initialCPTsITI(agentActions, agentVocab, bestPInfos, priorJointParams, pseudoCountSize)

    // 4. Convert county DBNs into parametery dbns
    val agentDBNs = cptsITIToDBNs(bestPInfos, cptsITI, pseudoCountSize)

    val initARewardScope = setOf(trueMDP.rewardScope.random())
    var agentRewardITI : RewardTree = RLeaf(initARewardScope, initARewardScope, createStats(initARewardScope, emptyList()), emptyList())
    var agentRewardDT = convertFromITI(agentRewardITI)
    var agentValueTree = agentRewardDT
    var agentPolicy = choosePolicy(agentDBNs.mapValues { (_, dbn) -> regress(agentValueTree, agentRewardDT, dbn) })
    val existRewardStates : MutableList<Trial> = ArrayList()

    var previousState = generateSample(trueMDP.vocab.toList())

    val lastStructUpdate = agentActions.associate { Pair(it, 0) }.toMutableMap()
    for(timeStep in 0..999){
        // e greedy choice strategy at the moment
        val projectedPrevState = project(previousState, agentVocab)
        val chosenAction = if(Math.random() < 0.9) matchLeaf(agentPolicy, previousState).value else agentActions.toList().random()

        // Remember to project this onto agent's known vocabulary
        val newTrueState = generateSample(trueMDP.dbns[chosenAction]!!, previousState)
        val projectedNewState = project(newTrueState, agentVocab)
        val reward = matchLeaf(trueMDP.rewardTree, newTrueState).value
        val agentSeqTrial = SequentialTrial(projectedPrevState, chosenAction, projectedNewState, reward)

        agentTrialHist[chosenAction]!!.add(agentSeqTrial)

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
                agentRewardITI = addAdditionalVocab(agentRewardITI, newRewardVars)
                agentVocab += newRewardVars
            }

            agentRewardITI = incrementalUpdate(agentRewardITI, Trial(projectedNewState, reward))
            agentRewardDT = convertFromITI(agentRewardITI)
        }

        if(oldVocab != agentVocab){
            // Do a structural update with old vocab to get most up-to-date reasonable parent set
            lastStructUpdate.mapValues { timeStep }
            for((a, rpMap) in reasonableParents){
                reasonableParents[a] = rpMap.mapValues { (rv, _) -> structuralUpdate(rv, pSetPriors[a]!![rv]!!, agentVocab, agentTrialHist[a]!!, expertEv, priorJointParams[a]!![rv]!!, aliveThresh, pseudoCountSize) }
                bestPInfos[a] = bestParents(reasonableParents[a]!!)
                cptsITI[a] = cptsITI[a]!!.mapValues { (rv, cpt) ->
                    changeVocab(cpt, bestPInfos[a]!![rv]!!.parentSet)
                }

                // Also, remake your dbn at this point (decision trees and all):
                agentDBNs[a] = cptsITI[a]!!.mapValues { (rv, cpt) ->
                    convertToCPT(cpt, bestPInfos[chosenAction]!![rv]!!.priorParams, pseudoCountSize)
                }

                priorJointParams[a] = cptsITI[a]!!.mapValues { (rv, cpt) ->
                    convertToJointProbTree(cpt, priorJointParams[a]!![rv]!!, pseudoCountSize)
                }

                // Do your "posterior to prior" stuff: I.e create new prior based on prev reasonable parents plus addition of new vocab
                val parentExtensionResult = posteriorToPrior(reasonableParents[a]!!, agentVocab, agentVocab - oldVocab, priorJointParams[a]!!, singleParentProb, aliveThresh, pseudoCountSize)
                pSetPriors[a] = parentExtensionResult.logPriors
                reasonableParents[a] = parentExtensionResult.reasonableParents

                // Throw away old trial counts
                agentTrialHist[a]!!.clear()

                bestPInfos[a] = bestParents(reasonableParents[a]!!)
                cptsITI[a] = initialCPTsITI(agentVocab, bestPInfos[a]!!, priorJointParams[a]!!, pseudoCountSize)
                agentDBNs[a] = cptsITI[a]!!.mapValues { (rv, cpt) -> convertToCPT(cpt, bestPInfos[a]!![rv]!!.priorParams, pseudoCountSize) }
            }
        }
        else{
            // Belief Function Update
            // 1. On relevant action dbn, add trial via parameter update
            var newReasonableParents = reasonableParents[chosenAction]!!.mapValues { (rv, rps) -> parameterUpdate(rv, rps, agentSeqTrial, expertEv, pseudoCountSize ) }

            // 2. If relevant condition triggered for structural update (empty reasonable parents, all reasonable zero, or 100 stateHist passed), do structural update
            // Why do we count interval over all variables? Surely each one individually would allow for greater control / less structural updates
            if(agentTrialHist[chosenAction]!!.size - lastStructUpdate[chosenAction]!! > 50){
                lastStructUpdate[chosenAction] = timeStep
                newReasonableParents = newReasonableParents.mapValues { (rv, _) -> structuralUpdate(rv, pSetPriors[chosenAction]!![rv]!!, agentVocab, agentTrialHist[chosenAction]!!, expertEv, priorJointParams[chosenAction]!![rv]!!, aliveThresh, pseudoCountSize) }
            }
            reasonableParents[chosenAction] = newReasonableParents

            // 3. Regardless of pset update, need to choose new most likely structure, check for consistency, and (if all fine, proceed to construct new cpd)
            bestPInfos[chosenAction] = bestParents(reasonableParents[chosenAction]!!)

            // 3 (b) (Consider structural constraints (what will you do if they are violated?))
            val structuralConstraintsHold = true // "Useless-vars?" function
            if(!structuralConstraintsHold){
                throw NotImplementedError("Haven't figured out how to handle this yet. Ask structure question? Ignore? Email Supervisor")
            }
        }

        cptsITI[chosenAction] = cptsITI[chosenAction]!!.mapValues { (rv, cpt) ->
            val newVocabDT = changeVocab(cpt, bestPInfos[chosenAction]!![rv]!!.parentSet)
            incrementalUpdate(newVocabDT, listOf(agentSeqTrial))
        }

        agentDBNs[chosenAction] = cptsITIToDBN(bestPInfos[chosenAction]!!, cptsITI[chosenAction]!!, pseudoCountSize)


        expert.agentOptimalActionHistory.add(wasActionOptimal(previousState, chosenAction, trueQs))
        // If agent did suboptimal action in previous state
        // And has acted suboptimally a sufficient proprortion of times in the past
        if(timeStep - expert.lastAdvice < expert.adviceInterval && !expert.agentOptimalActionHistory.last() && poorRecentPerformance(expert.agentOptimalActionHistory, expert.historyWindow, expert.mistakeProportion)){
            // Then tell the agent about the better action
            val betterAction = betterActionAdvice(previousState, expert)

            if(betterAction !in agentActions){
                agentActions.add(betterAction)

                priorJointParams[betterAction] = unifStartIDTransJointDBN(agentVocab, 0.1)
                agentTrialHist[betterAction] = ArrayList()
                pSetPriors[betterAction] = initialPSetPriors(agentVocab, singleParentProb)
                reasonableParents[betterAction] = structuralUpdate(agentVocab, agentTrialHist[betterAction]!!, expertEv, pSetPriors[betterAction]!!, aliveThresh, priorJointParams[betterAction]!!, pseudoCountSize)
                bestPInfos[betterAction] = bestParents(reasonableParents[betterAction]!!)
                cptsITI[betterAction] = initialCPTsITI(agentVocab, bestPInfos[betterAction]!!, priorJointParams[betterAction]!!, pseudoCountSize)
                agentDBNs[betterAction] = cptsITIToDBN(bestPInfos[betterAction]!!, cptsITI[betterAction]!!, pseudoCountSize)
            }

            val (newValue, _) = applyExpertAdvice(agentRewardDT, agentValueTree, agentDBNs, Pair(projectedPrevState, betterAction))
            agentValueTree = newValue


            // Resolving misunderstandings
            if(!actionAdvice.containsKey(projectedPrevState) || actionAdvice[projectedPrevState]!!.second == betterAction){
                actionAdvice[projectedPrevState] = Pair(prevStateTS, betterAction)
            }
            else{
                val earlierAdvice = actionAdvice[projectedPrevState]!!
                val latestAdvice = Pair(prevStateTS, betterAction)
                val resolution = resolveMisunderstanding(earlierAdvice.first, latestAdvice.first, expert)
                val resolutionVar = resolution.firstAssignment.first
                if(resolutionVar in agentVocab){
                    throw IllegalStateException("If differing variable was already in agent vocab, then why was there a misunderstanding?")
                }
                actionAdvice.clear()
                actionAdvice[projectedPrevState + resolution.firstAssignment] = earlierAdvice
                actionAdvice[projectedPrevState + resolution.secondAssignment] = latestAdvice

                agentVocab += resolutionVar
                // Add new belief variable
                // This whole part is probably extractable into a function
                // Do a structural update with old vocab to get most up-to-date reasonable parent set
                lastStructUpdate.mapValues { timeStep }
                for((a, rpMap) in reasonableParents){
                    reasonableParents[a] = rpMap.mapValues { (rv, _) -> structuralUpdate(rv, pSetPriors[a]!![rv]!!, agentVocab, agentTrialHist[a]!!, expertEv, priorJointParams[a]!![rv]!!, aliveThresh, pseudoCountSize) }
                    bestPInfos[a] = bestParents(reasonableParents[a]!!)
                    cptsITI[a] = cptsITI[a]!!.mapValues { (rv, cpt) ->
                        changeVocab(cpt, bestPInfos[a]!![rv]!!.parentSet)
                    }

                    // Also, remake your dbn at this point (decision trees and all):
                    agentDBNs[a] = cptsITI[a]!!.mapValues { (rv, cpt) ->
                        convertToCPT(cpt, bestPInfos[chosenAction]!![rv]!!.priorParams, pseudoCountSize)
                    }

                    priorJointParams[a] = cptsITI[a]!!.mapValues { (rv, cpt) ->
                        convertToJointProbTree(cpt, priorJointParams[a]!![rv]!!, pseudoCountSize)
                    }

                    // Do your "posterior to prior" stuff: I.e create new prior based on prev reasonable parents plus addition of new vocab
                    val parentExtensionResult = posteriorToPrior(reasonableParents[a]!!, agentVocab, setOf(resolutionVar), priorJointParams[a]!!, singleParentProb, aliveThresh, pseudoCountSize)
                    pSetPriors[a] = parentExtensionResult.logPriors
                    reasonableParents[a] = parentExtensionResult.reasonableParents

                    // Throw away old trial counts
                    agentTrialHist[a]!!.clear()

                    bestPInfos[a] = bestParents(reasonableParents[a]!!)
                    cptsITI[a] = initialCPTsITI(agentVocab, bestPInfos[a]!!, priorJointParams[a]!!, pseudoCountSize)
                    agentDBNs[a] = cptsITI[a]!!.mapValues { (rv, cpt) -> convertToCPT(cpt, bestPInfos[a]!![rv]!!.priorParams, pseudoCountSize) }
                }
            }
        }

        // Policy Updates
        val (newValue, newQTrees) = prune(incrementalSVI(agentRewardDT, agentValueTree, agentDBNs))
        agentValueTree = newValue
        agentPolicy = choosePolicy(newQTrees)

        previousState = newTrueState

        // Remember to bin all your inferences when discovering new belief vocab
    }
}


private fun cptsITIToDBNs(bestPInfos: Map<Action, Map<RandomVariable, SeqPInfo>>, cptsITI: Map<Action, Map<RandomVariable, ProbTree>>, pseudoCountSize: Double): MutableMap<String, DynamicBayesNet> {
    return cptsITI
        .mapValues { (a, cpts) -> cptsITIToDBN(bestPInfos[a]!!, cpts, pseudoCountSize) }
        .toMutableMap()
}

fun cptsITIToDBN(bestPInfos : Map<RandomVariable, SeqPInfo>, cptsITI: Map<RandomVariable, ProbTree>, pseudoCountSize: Double) =
    cptsITI.mapValues { (rv, dt) -> convertToCPT(dt, bestPInfos[rv]!!.priorParams, pseudoCountSize)}

private fun initialCPTsITI(agentActions: MutableSet<String>, agentVocab: Set<RandomVariable>, bestPInfos: MutableMap<Action, Map<RandomVariable, SeqPInfo>>, priorDBNs: MutableMap<String, DynamicBayesNet>, pseudoCountSize: Double): MutableMap<Action, Map<RandomVariable, ProbTree>> =
    agentActions
        .associate { Pair(it, initialCPTsITI(agentVocab, bestPInfos[it]!!, priorDBNs[it]!!, pseudoCountSize)) }
        .toMutableMap()

fun initialCPTsITI(agentVocab: Set<RandomVariable>, bestPInfos: Map<RandomVariable, SeqPInfo>, priorDBN: DynamicBayesNet, pseudoCountSize: Double): Map<RandomVariable, ProbTree> =
    agentVocab.associate { rv ->
        val bestPSet = bestPInfos[rv]!!.parentSet
        Pair(rv, emptyPLeaf(rv, bestPSet, priorDBN[rv]!!, pseudoCountSize))
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

/*
data class Agent(
    val vocab : Set<RandomVariable>,
    val trials : Map<Action, SequentialTrial>,
    val priorDBN : DynamicBayesNet,
    val reasonableParents : Map<Action, List<SeqPInfo>>,
    val pSetPriors : Map<Action, Map<RandomVariable, LogPrior<PSet>>>,
    val dbnITI : Map<Action, Map<RandomVariable, ProbTree>>,
    val dbns : Map<Action, DynamicBayesNet>,
    val rewardITI : RewardTree,
    val reward : DecisionTree<Reward>,
    val valueTree : DecisionTree<Double>,
    val policy : DecisionTree<Action>
)
*/
