import Utils.project
import Utils.random
import java.io.FileWriter
import java.io.PrintWriter

fun main(args : Array<String>){
    val mdp = loadMDP("mdps/coffee.json")

    /*
    val configs = listOf(
        ExperimentConfig("RandomPolicy", 1.0, false, 10, 0.01),
        ExperimentConfig("NoExpert", 0.1, false, 10000000, 0.01),
        ExperimentConfig("TruePolicy", 0.1, true, 10, 0.01),
        ExperimentConfig("NoPruning", 0.1, false, 10, 0.00),
        ExperimentConfig("Default", 0.1, false, 10, 0.01)
    )

    for(config in configs){
        for(i in 0..30){
            ResultsLogger.filePath = "logs/coffeeTest${config.experimentName}-$i"
            runLearning(mdp, config)
        }
    }
    */

    val config = ExperimentConfig("Default", 0.1, false, 10, 0.01)
    ResultsLogger.filePath = "logs/coffeeTest${config.experimentName}-1"
    runLearning(mdp, config)
}

data class ExperimentConfig(
    val experimentName : String,
    val exploreAmount: Double,
    val useTrue : Boolean,
    val expertAdviceInterval : Int,
    val pruneThreshold : Double
)


object ResultsLogger{
    val printToConsole = true
    var filePath: String = "empty"
    fun logTimeStampedResult(timeStamp: TimeStamp, result : Double, metric : String){
        if(printToConsole){
            println("$metric - ts: $timeStamp, val: $result")
        }
        val writer = PrintWriter(FileWriter("$filePath-$metric.txt", true))
        writer.println("$timeStamp, $result")
        writer.close()
    }

    fun logBetterActionAdvice(timeStamp: TimeStamp, agentAction : Action, betterAction: Action, state: RVAssignment){
        if(printToConsole){
            println("Expert Advised $betterAction over agent action $agentAction at time $timeStamp in state $state")
        }
        val writer = PrintWriter(FileWriter("$filePath-expertBetterAction.txt", true))
        writer.println("$timeStamp, $agentAction, $betterAction")
        writer.close()
    }

    fun logJSONStructure(obj : Any, objectName : String){
        saveToJson(obj, "$filePath-$objectName")
    }
}

fun runLearning(trueMDP: MDP, experimentConfig: ExperimentConfig){
    val (truePolicy, trueValues) = structuredValueIteration(trueMDP.rewardTree, trueMDP.dbns)
    val trueQs = trueMDP.dbns.mapValues { (_, dbn) -> regress(trueValues, trueMDP.rewardTree, dbn)}
    val expert = Expert(experimentConfig.expertAdviceInterval, 0.25, experimentConfig.expertAdviceInterval, HashSet(), trueMDP, trueQs)

    val aliveThresh = 0.01
    val singleParentProb = 0.1
    val pseudoCountSize = 1.0
    val buntineUpdater = BuntineUpdater(aliveThresh, pseudoCountSize, singleParentProb)

    var agentVocab = trueMDP.vocab.toSet()
    val agentActions = trueMDP.actions.toMutableSet()
    val agentTrialHist = agentActions.associate { Pair(it, ArrayList<Pair<TimeStamp, SequentialTrial>>()) }.toMutableMap()
    val expertEv = emptyList<DirEdge>()
    val actionAdvice = HashMap<RVAssignment, Pair<TimeStamp, Action>>()

    val agentDBNInfos = agentActions.associate{ Pair(it, buntineUpdater.initialDBNInfo(agentVocab, expertEv, 0))}.toMutableMap()


    val initARewardScope = setOf(trueMDP.rewardScope.random())
    var agentRewardITI = emptyRewardTree(initARewardScope)
    var agentRewardDT = convertFromITI(agentRewardITI.first)
    var agentValueTree = agentRewardDT
    var agentPolicy = choosePolicy(agentDBNInfos.mapValues { (_, dbnInfo) -> regress(agentValueTree, agentRewardDT, dbnInfo.dbn) })
    val existRewardStates = ArrayList<Trial>()

    var previousState = generateInitialState(trueMDP.vocab.toList(), trueMDP.terminalDescriptions)

    var tStep = 0
    while(tStep < 4000){
        println(tStep)
        val projectedPrevState = project(previousState, agentVocab)

        val chosenAction = if(experimentConfig.useTrue){
            eGreedy(agentActions, truePolicy, previousState, experimentConfig.exploreAmount)
        }
        else{
            eGreedy(agentActions, agentPolicy, previousState, experimentConfig.exploreAmount)
        }

        println("Previous State: $previousState")
        println("Chosen Action: $chosenAction")

        // Remember to project this onto agent's known vocabulary
        val newTrueState = generateSample(trueMDP.dbns[chosenAction]!!, previousState)
        val projectedNewState = project(newTrueState, agentVocab)
        val reward = matchLeaf(trueMDP.rewardTree, newTrueState).value

        println("New State : $newTrueState, Reward : $reward")
        ResultsLogger.logTimeStampedResult(tStep, reward, "reward")

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
                agentRewardITI = agentRewardITI.copy(first = removeExample(agentRewardITI.first, redundantTrial, agentRewardITI.second))
            }

            existRewardStates.add(rewardTrial)

            // Consistency Check
            val solvableReward = unfactoredReward(agentRewardITI.second.vocab.toList() , existRewardStates)
            if(solvableReward == null){
                val newRewardVars = whatsInRewardScope(agentRewardITI.second.vocab, expert)
                tStep += 1
                agentVocab += newRewardVars
                agentRewardITI = changeAllowedVocab(agentRewardITI, agentVocab)
            }

            agentRewardITI = agentRewardITI.copy(first = incrementalUpdate(agentRewardITI.first, listOf(Trial(projectedNewState, reward)), agentRewardITI.second))
            agentRewardDT = convertFromITI(agentRewardITI.first)
        }

        // DBN Updates
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
        if(tStep - expert.lastAdvice > expert.adviceInterval && !expert.agentOptimalActionHistory.last() && poorRecentPerformance(expert.agentOptimalActionHistory, expert.historyWindow, expert.mistakeProportion)){
            // Then tell the agent about the better action
            tStep += 1
            expert.lastAdvice = tStep
            val betterAction = betterActionAdvice(previousState, expert)

            ResultsLogger.logBetterActionAdvice(tStep, chosenAction, betterAction, previousState)

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
        val (newValue, newQTrees) = incrementalSVI(agentRewardDT, agentValueTree, agentDBNInfos.mapValues {it.value.dbn}, experimentConfig.pruneThreshold)
        agentValueTree = newValue
        agentPolicy = choosePolicy(newQTrees)

        if(trueMDP.terminalDescriptions.any{ partialMatch(it, newTrueState) }){
            println("Terminal State Hit! Resetting...")
            previousState = generateInitialState(trueMDP.vocab.toList(), trueMDP.terminalDescriptions)
        }
        else{
            previousState = newTrueState
        }

        tStep += 1
        println()
    }

    ResultsLogger.logJSONStructure(agentPolicy, "finalPolicy")

    for((action, dbnInfo) in agentDBNInfos){
        val rawStruct = dbnInfo.bestPInfos
            .mapKeys { it.key.name }
            .mapValues { it.value.parentSet.map { it.name } }
        ResultsLogger.logJSONStructure(rawStruct, "transition-$action")
    }

    println("Done")
}

fun eGreedy(actions : Set<Action>, exploitPolicy : PolicyTree, state : RVAssignment, exploreAmount : Double) : Action {
    if(Math.random() < 1 - exploreAmount){
        return matchLeaf(exploitPolicy, state).value
    }
    else{
        println("Random Exploration")
        return actions.random()
    }
}

fun cptsITIToDBN(cptsITI: Map<RandomVariable, ProbTree>, jointParamPriors : Map<RandomVariable, DecisionTree<Factor>>, pseudoCountSize: Double) =
    cptsITI.mapValues { (rv, pt) -> convertToCPT(rv, pt.first, jointParamPriors[rv]!!, pseudoCountSize)}


fun initialCPTsITI(agentVocab: Set<RandomVariable>, bestPInfos: Map<RandomVariable, SeqPInfo>, priorDBN: DynamicBayesNet, pseudoCountSize: Double): Map<RandomVariable, ProbTree> =
    agentVocab.associate { rv ->
        val bestPSet = bestPInfos[rv]!!.parentSet
        Pair(rv, emptyProbTree(rv, bestPSet, pseudoCountSize, 0.7))
    }

fun initialPSetPriors(beliefVocab: Set<RandomVariable>, singleParentProb: Double) : Map<RandomVariable, LogPrior<PSet>> =
    beliefVocab.associate { rv ->
        Pair(rv, minParentsPrior(beliefVocab, singleParentProb))
    }



