import Utils.project
import Utils.random
import org.apache.commons.io.FilenameUtils
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter

fun main(args : Array<String>){

    if(args.size != 3){
        throw IllegalArgumentException("3 Arguments Required: <mdp-file-path> <config-file-path> <experiment-id>")
    }

    val mdpPath = args[0]
    val mdpName = FilenameUtils.getBaseName(mdpPath)
    val mdp = loadMDP(mdpPath)

    val configPath = args[1]
    val configName = FilenameUtils.getBaseName(configPath)
    val config = loadJson(configPath, ExperimentConfig::class.java)

    val logPath = "logs/$mdpName-$configName"
    val taskId = args[2]

    ResultsLogger.setPath(logPath, taskId)
    runLearning(mdp, config)
}

data class ExperimentConfig(
    val experimentName : String,
    val numTrials : Int,
    val exploreAmount: Double,
    val useTrue : Boolean,
    val expertAdviceInterval : Int,
    val prune: Boolean,
    val pruneRange: Double,
    val dbnUpdater : DBNUpdater
)


object ResultsLogger{
    val printToConsole = true
    private var outputFolder : String? = null
    private var fileName: String? = null

    fun setPath(folderPath: String, fName : String){
        outputFolder = folderPath
        fileName = fName
        File(outputFolder).mkdirs()
    }

    fun logTimeStampedResult(timeStamp: TimeStamp, result : Double, metric : String){
        if(printToConsole){
            println("$metric - ts: $timeStamp, val: $result")
        }
        val writer = PrintWriter(FileWriter("${outputFolder!!}/${fileName!!}-$metric.txt", true))
        writer.println("$timeStamp, $result")
        writer.close()
    }

    fun logBetterActionAdvice(timeStamp: TimeStamp, agentAction : Action, betterAction: Action, state: RVAssignment){
        if(printToConsole){
            println("Expert Advised $betterAction over agent action $agentAction at time $timeStamp in state $state")
        }
        val writer = PrintWriter(FileWriter("${outputFolder!!}/${fileName!!}-expertBetterAction.txt", true))
        writer.println("$timeStamp, $agentAction, $betterAction")
        writer.close()
    }

    fun logJSONStructure(obj : Any, objectName : String){
        saveToJson(obj, "${outputFolder!!}/${fileName!!}-$objectName")
    }
}

fun nextState(proposedNextState : RVAssignment, trueMDP : MDP) : RVAssignment{
    if(trueMDP.terminalDescriptions.any{ partialMatch(it, proposedNextState) }){
            println("Terminal State Hit! Resetting...")
            return generateInitialState(trueMDP.vocab.toList(), trueMDP.startStateDescriptions, trueMDP.terminalDescriptions)
        }
        else{
            return proposedNextState
        }
}

fun runLearning(trueMDP: MDP, config: ExperimentConfig){
    val (truePolicy, trueValues) = structuredValueIteration(trueMDP.rewardTree, trueMDP.dbns, trueMDP.vocab.toList(), config.pruneRange)
    val truePolicySize = numLeaves(truePolicy)
    val trueValueSize = numLeaves(trueValues)
    val trueQs = trueMDP.dbns.mapValues { (_, dbn) -> regress(trueValues, trueMDP.rewardTree, dbn)}
    val expert = Expert(config.expertAdviceInterval, 0.25, config.expertAdviceInterval, HashSet(), trueMDP, trueQs)

    var agentVocab = trueMDP.vocab.toSet()
    val agentActions = trueMDP.actions.toMutableSet()
    val agentTrialHist = agentActions.associate { Pair(it, ArrayList<Pair<TimeStamp, SequentialTrial>>()) }.toMutableMap()
    val expertEv = emptyList<DirEdge>()
    val actionAdvice = HashMap<RVAssignment, Pair<TimeStamp, Action>>()

    val agentDBNInfos = agentActions.associate{ Pair(it, config.dbnUpdater.initialDBNInfo(agentVocab, expertEv, 0))}.toMutableMap()


    val initARewardScope = setOf(trueMDP.rewardScope.random())
    var agentRewardITI = emptyRewardTree(initARewardScope)
    var agentRewardDT = convertFromITI(agentRewardITI.first)
    var agentRangedValueTree : DecisionTree<Range> =  toRanged(agentRewardDT)
    var agentPolicy = choosePolicy(agentDBNInfos.mapValues { (_, dbnInfo) -> regressRanged(agentRangedValueTree, agentRewardDT, dbnInfo.dbn) }, agentVocab.toList())
    val existRewardStates = ArrayList<Trial>()

    var previousState = generateInitialState(trueMDP.vocab.toList(), trueMDP.startStateDescriptions, trueMDP.terminalDescriptions)

    var tStep = 0
    while(tStep < config.numTrials){
        println(tStep)
        val projectedPrevState = project(previousState, agentVocab)

        val chosenAction = if(config.useTrue){
            eGreedyWithAdvice(agentActions, truePolicy, actionAdvice, previousState, config.exploreAmount)
        }
        else{
            eGreedyWithAdvice(agentActions, agentPolicy, actionAdvice, previousState, config.exploreAmount)
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
                println("Expert told agent about new reward vars: ${newRewardVars}")
                tStep += 1
                agentVocab += newRewardVars
                agentRewardITI = changeAllowedVocab(agentRewardITI, agentRewardITI.second.vocab + newRewardVars)
            }

            agentRewardITI = agentRewardITI.copy(first = incrementalUpdate(agentRewardITI.first, listOf(Trial(projectedNewState, reward)), agentRewardITI.second))
            agentRewardDT = convertFromITI(agentRewardITI.first)
        }

        // DBN Updates
        if(oldVocab != agentVocab){
            agentDBNInfos.mapValues { (a, dbnInfo) -> config.dbnUpdater.addBeliefVariables(agentVocab - oldVocab, tStep, agentTrialHist[a]!!.map{it.second}, expertEv, dbnInfo) }
            agentTrialHist.forEach{ _, trials -> trials.clear() }
            actionAdvice.clear()
        }
        else{
            agentDBNInfos[chosenAction] = config.dbnUpdater.trialUpdate(agentSeqTrial.second, agentTrialHist[chosenAction]!!.map{it.second}, expertEv, tStep, agentDBNInfos[chosenAction]!!)
        }


        // Better Action Advice
        /*
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
        */

        // Policy Updates
        if(!config.useTrue){
            val (newValue, newQTrees) = incrementalSVI(agentRewardDT, agentRangedValueTree, agentDBNInfos.mapValues {it.value.dbn}, agentVocab.toList(), config.pruneRange)
            agentRangedValueTree = newValue
            agentPolicy = choosePolicy(newQTrees, agentVocab.toList())
        }

        logNodeSizes(agentDBNInfos.mapValues { it.value.dbn }, fromRanged(agentRangedValueTree), tStep)

        previousState = nextState(newTrueState, trueMDP)
        tStep += 1
        println()
    }

    if(config.useTrue){
        ResultsLogger.logJSONStructure(truePolicy, "finalPolicy")
    }
    else{
        ResultsLogger.logJSONStructure(agentPolicy, "finalPolicy")
    }

    agentDBNInfos.forEach { action, dbnInfo -> ResultsLogger.logJSONStructure(rawStructData(dbnInfo), "transition-$action") }
    println("Done")
}

fun logNodeSizes(dbns : Map<Action, Map<RandomVariable, DecisionTree<Factor>>>, valueTree : DecisionTree<Double>, tStep : Int){
    val transitionNodes = dbns.values.sumBy { it.entries.sumBy { (_, dbn) -> numNodes(dbn) } }.toDouble()
    val valueNodes = numNodes(valueTree).toDouble()
    ResultsLogger.logTimeStampedResult(tStep, transitionNodes, "TransitionNodes")
    ResultsLogger.logTimeStampedResult(tStep, valueNodes, "ValueNodes")
}

fun eGreedyWithAdvice(actions : Set<Action>, exploitPolicy : PolicyTree, expertAdvice : Map<RVAssignment, Pair<TimeStamp, Action>>, state : RVAssignment, exploreAmount : Double) : Action {
    if(Math.random() < 1 - exploreAmount){
        if(state in expertAdvice){
            return expertAdvice[state]!!.second
        }

        return matchLeaf(exploitPolicy, state).value
    }
    else{
        println("Random Exploration")
        return actions.random()
    }
}

fun cptsITIToDBN(cptsITI: Map<RandomVariable, ProbTree>, jointParamPriors : Map<RandomVariable, DecisionTree<Factor>>, pseudoCountSize: Double) =
    cptsITI.mapValues { (rv, pt) -> convertToCPT(rv, pt.first, jointParamPriors[rv]!!, pseudoCountSize)}


fun initialCPTsITI(agentVocab: Set<RandomVariable>, bestPSets: Map<RandomVariable, PSet>, pseudoCountSize: Double): Map<RandomVariable, ProbTree> =
    agentVocab.associate { rv ->
        Pair(rv, emptyProbTree(rv, bestPSets[rv]!!, pseudoCountSize, 0.7))
    }

fun initialPSetPriors(beliefVocab: Set<RandomVariable>, singleParentProb: Double) : Map<RandomVariable, LogPrior<PSet>> =
    beliefVocab.associate { rv ->
        Pair(rv, minParentsPrior(rv, beliefVocab, singleParentProb))
    }



