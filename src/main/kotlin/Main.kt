import Utils.project
import Utils.random
import org.apache.commons.io.FilenameUtils
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter

fun main(args : Array<String>){

    if(args.size != 4){
        throw IllegalArgumentException("4 Arguments Required: <mdp-file> <awareness-file> <config-file> <experiment-id>")
    }

    val mdpPath = FilenameUtils.concat("mdps", args[0])
    val mdpName = FilenameUtils.getBaseName(mdpPath)
    val mdp = loadMDP(mdpPath)

    val startingAwarenessPath = FilenameUtils.concat("startMDPs", args[1])
    val startingAwarenessName = FilenameUtils.getBaseName(startingAwarenessPath)
    val startingAwareness = loadJson(startingAwarenessPath, MDPAwareness::class)


    val configPath = FilenameUtils.concat("configs", args[2])
    val configName = FilenameUtils.getBaseName(configPath)
    val config = loadJson(configPath, ExperimentConfig::class)

    val logPath = "logs/$mdpName-$startingAwarenessName-$configName"
    val taskId = args[3]

    ResultsLogger.setPath(logPath, taskId)
    runLearning(mdp, startingAwareness, config)
}

data class ExperimentConfig(
    val experimentName : String,
    val numTrials : Int,
    val exploreAmount: Double,
    val useTrue : Boolean,
    val expertAdviceInterval : Int,
    val policyErrorTolerance : Double,
    val maxEpisodeLength : Int,
    val prune: Boolean,
    val pruneRange: Double,
    val conserveVals : Boolean,
    val dbnUpdater : DBNUpdater
)

data class MDPAwareness(val variables : Set<String>, val actions : Set<Action>, val rewardDomain : Set<String>)


object ResultsLogger{
    var printToConsole = true
    private var outputFolder : String? = null
    private var fileName: String? = null

    fun setPath(folderPath: String, fName : String){
        outputFolder = folderPath
        fileName = fName
        File(outputFolder).mkdirs()
        File(outputFolder).listFiles()
            .filter { it.name.startsWith(fName) }
            .forEach { it.delete() }
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
        writer.println("$timeStamp, $state, $agentAction, $betterAction")
        writer.close()
    }

    fun logBeliefMisunderstandingAdvice(earlierTime : TimeStamp, currentTime : TimeStamp, beliefVocabResolver : RandomVariable){
        if(printToConsole){
            println("Expert Resolved Misunderstanding at times $earlierTime, and $currentTime by mentioning $beliefVocabResolver")
        }
        val writer = PrintWriter(FileWriter("${outputFolder!!}/${fileName!!}-expertMisunderstanding.txt", true))
        writer.println("$earlierTime, $currentTime, $beliefVocabResolver")
        writer.close()
    }

    fun logUnexpectedRewardAdvice(timeStamp: TimeStamp, domainAdvice : Set<RandomVariable>){
        if(printToConsole){
            println("Expert Mentioned Reward Variables $domainAdvice at time $timeStamp")
        }
        val writer = PrintWriter(FileWriter("${outputFolder!!}/${fileName!!}-expertUnexpectedReward.txt", true))
        writer.println("$timeStamp, $domainAdvice")
        writer.close()
    }

    fun logJSONStructure(obj : Any, objectName : String){
        saveToJson(obj, "${outputFolder!!}/${fileName!!}-$objectName")
    }
}



fun runLearning(trueMDP: MDP, startingAwareness : MDPAwareness, config: ExperimentConfig){
    val (truePolicy, trueValues) = structuredValueIteration(trueMDP.rewardTree, trueMDP.dbns, trueMDP.vocab.toList(), trueMDP.terminalDescriptions, trueMDP.discount, config.pruneRange)
    val truePolicySize = numLeaves(truePolicy)
    val trueValueSize = numLeaves(trueValues)
    val startStates = allAssignments(trueMDP.vocab.toList())
        .filter { trueMDP.startStateDescriptions.isEmpty() || trueMDP.startStateDescriptions.any { startDesc -> partialMatch(startDesc, it) } }

    val trueQs = trueMDP.dbns.mapValues { (_, dbn) -> regressRanged(trueValues, trueMDP.rewardTree, dbn, trueMDP.terminalDescriptions, trueMDP.discount)}
    val expert = Expert(config.expertAdviceInterval, config.policyErrorTolerance, config.expertAdviceInterval, config.maxEpisodeLength, HashSet(), trueMDP, trueQs.mapValues{ fromRanged(it.value) })

    var agentVocab = trueMDP.vocab.asSequence().filter { it.name in startingAwareness.variables }.toSet()
    val agentActions = trueMDP.actions.asSequence().filter { it in startingAwareness.actions }.toMutableSet()
    val agentTrialHist = agentActions.associate { Pair(it, ArrayList<Pair<TimeStamp, SequentialTrial>>()) }.toMutableMap()
    val expertEv = emptyList<DirEdge>()
    val actionAdvice = HashMap<RVAssignment, Pair<TimeStamp, Action>>()

    val agentDBNInfos = agentActions.associate{ Pair(it, config.dbnUpdater.initialDBNInfo(agentVocab, expertEv, 0))}.toMutableMap()


    val initARewardScope = trueMDP.rewardScope.asSequence().filter { it.name in startingAwareness.rewardDomain }.toSet()
    var agentRewardITI = emptyRewardTree(initARewardScope)
    var agentRewardDT = convertFromITI(agentRewardITI.first)
    var agentRangedValueTree : DecisionTree<Range> =  toRanged(agentRewardDT)
    var agentPolicy = choosePolicy(agentDBNInfos.mapValues { (_, dbnInfo) -> regressRanged(agentRangedValueTree, agentRewardDT, dbnInfo.dbn, trueMDP.terminalDescriptions, trueMDP.discount) }, agentVocab.toList())
    val existRewardStates = ArrayList<Trial>()

    var previousState = generateInitialState(trueMDP.vocab.toList(), trueMDP.startStateDescriptions, trueMDP.terminalDescriptions)

    for(tStep in 0 until config.numTrials){
        println(tStep)
        val projectedPrevState = project(previousState, agentVocab)

        val chosenAction = if(config.useTrue){
            eGreedyWithAdvice(agentActions, truePolicy, actionAdvice, projectedPrevState, config.exploreAmount)
        }
        else{
            eGreedyWithAdvice(agentActions, agentPolicy, actionAdvice, projectedPrevState, config.exploreAmount)
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
        expert.episodeHistory.last().add(SequentialTrial(previousState, chosenAction, newTrueState, reward))

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
                ResultsLogger.logUnexpectedRewardAdvice(tStep, newRewardVars)
                agentVocab += newRewardVars
                agentRewardITI = changeAllowedVocab(agentRewardITI, agentRewardITI.second.vocab + newRewardVars)
            }

            agentRewardITI = agentRewardITI.copy(first = incrementalUpdate(agentRewardITI.first, listOf(Trial(projectedNewState, reward)), agentRewardITI.second))
            agentRewardDT = convertFromITI(agentRewardITI.first)
        }

        // DBN Updates
        if(oldVocab != agentVocab){
            // Add the new belief variable to all previous DBNs, and wipe the trials
            for((a, dbnInfo) in agentDBNInfos){
                agentDBNInfos[a] = config.dbnUpdater.addBeliefVariables(agentVocab - oldVocab, tStep, agentTrialHist[a]!!.map{it.second}, expertEv, dbnInfo)
            }

            if(!config.conserveVals){
                agentRangedValueTree = toRanged(agentRewardDT)
            }

            agentTrialHist.forEach{ _, trials -> trials.clear() }
            actionAdvice.clear()
        }
        else{
            agentDBNInfos[chosenAction] = config.dbnUpdater.trialUpdate(agentSeqTrial.second, agentTrialHist[chosenAction]!!.map{it.second}, expertEv, tStep, agentDBNInfos[chosenAction]!!)
        }

        // Better Action Advice
        if(
            tStep - expert.lastAdvice > expert.adviceInterval &&
            !wasActionOptimal(previousState, chosenAction, trueQs.mapValues { minVals(it.value) }) &&
            (policyErrorEstimate(relevantEps(expert.episodeHistory, expert.lastAdvice), trueMDP.discount, trueMDP, startStates, minVals(trueValues)) > expert.policyErrorTolerance ||
                expert.episodeHistory.last().size > expert.maxEpisodeLength)){
            // Then tell the agent about the better action
            expert.lastAdvice = tStep
            val betterAction = betterActionAdvice(previousState, expert)

            ResultsLogger.logBetterActionAdvice(tStep, chosenAction, betterAction, previousState)

            // Create a brand new DBN if action was previously unknown
            if(betterAction !in agentActions){
                agentActions.add(betterAction)
                agentDBNInfos[betterAction] = config.dbnUpdater.initialDBNInfo(agentVocab, expertEv, tStep)
                agentTrialHist[betterAction] = ArrayList()

                if(!config.conserveVals){
                    agentRangedValueTree = toRanged(agentRewardDT)
                }
            }


            // Resolving misunderstandings
            if(!actionAdvice.containsKey(projectedPrevState) || actionAdvice[projectedPrevState]!!.second == betterAction){
                actionAdvice[projectedPrevState] = Pair(agentSeqTrial.first, betterAction)
            }
            else{
                val earlierAdvice = actionAdvice[projectedPrevState]!!
                val latestAdvice = Pair(agentSeqTrial.first, betterAction)
                val resolution = resolveMisunderstanding(earlierAdvice.first, latestAdvice.first, expert)

                val resolutionVar = resolution.firstAssignment.first
                if(resolutionVar in agentVocab){
                    throw IllegalStateException("If differing variable was already in agent vocab, then why was there a misunderstanding?")
                }

                ResultsLogger.logBeliefMisunderstandingAdvice(earlierAdvice.first, latestAdvice.first, resolutionVar)

                actionAdvice.clear()
                actionAdvice[projectedPrevState + resolution.firstAssignment] = earlierAdvice
                actionAdvice[projectedPrevState + resolution.secondAssignment] = latestAdvice

                agentVocab += resolutionVar

                // Add the new belief variable to all previous DBNs, and wipe the trials
                for((a, dbnInfo) in agentDBNInfos){
                    agentDBNInfos[a] = config.dbnUpdater.addBeliefVariables(setOf(resolutionVar), tStep, agentTrialHist[a]!!.map{it.second}, expertEv, dbnInfo)
                }
                agentTrialHist.forEach { _, trials -> trials.clear() }

                if(!config.conserveVals){
                    agentRangedValueTree = toRanged(agentRewardDT)
                }
            }
        }

        // Policy Updates
        if(!config.useTrue){
            val (newValue, newQTrees) = incrementalSVI(agentRewardDT, agentRangedValueTree,
                agentDBNInfos.mapValues {it.value.dbn}, agentVocab.toList(), trueMDP.terminalDescriptions, trueMDP.discount, config.pruneRange)
            agentRangedValueTree = newValue
            agentPolicy = choosePolicy(newQTrees, agentVocab.toList())
        }

        logNodeSizes(agentDBNInfos.mapValues { it.value.dbn }, fromRanged(agentRangedValueTree), tStep)
        ResultsLogger.logTimeStampedResult(tStep, agentVocab.size.toDouble(), "VocabSize")
        ResultsLogger.logTimeStampedResult(tStep, agentActions.size.toDouble(), "ActionsSize")

        previousState = if(isTerminal(newTrueState, trueMDP)){
            println("Terminal State Hit! Resetting...")
            expert.episodeHistory.add(mutableListOf())
            generateInitialState(trueMDP.vocab.toList(), trueMDP.startStateDescriptions, trueMDP.terminalDescriptions)
        } else{
            newTrueState
        }
        println()
    }

    val loggedPolicy = if(config.useTrue) truePolicy else agentPolicy
    ResultsLogger.logJSONStructure(loggedPolicy, "finalPolicy")

    // Log DBNs for each action
    agentDBNInfos.forEach { action, dbnInfo -> ResultsLogger.logJSONStructure(rawStructData(dbnInfo), "transition-$action") }
    val finalPolicyError = policyErrorEstimate(relevantEps(expert.episodeHistory, expert.lastAdvice), trueMDP.discount, trueMDP, startStates, minVals(trueValues))
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


/*
fun convertToCPTNoPrior(cptsITI : Map<RandomVariable, ProbTree>) =
    cptsITI.mapValues { (rv, pt) -> convertToCPTNoPrior(rv, pt.first) }
*/


fun initialCPTsITI(agentVocab: Set<RandomVariable>, bestPSets: Map<RandomVariable, PSet>, jointPriorParams : Map<RandomVariable, DecisionTree<Factor>>, pseudoCountSize: Double): Map<RandomVariable, ProbTree> =
    agentVocab.associate { rv ->
        Pair(rv, emptyProbTree(rv, bestPSets[rv]!!, jointPriorParams[rv]!!, pseudoCountSize, 0.7))
    }


fun initialPSetPriors(beliefVocab: Set<RandomVariable>, singleParentProb: Double) : Map<RandomVariable, LogPrior<PSet>> =
    beliefVocab.associate { rv ->
        Pair(rv, minParentsPrior(rv, beliefVocab, singleParentProb))
    }

/*
fun customPriorCPTsITI(agentVocab : Set<RandomVariable>, bestPSets: Map<RandomVariable, PSet>, jointParamPrior : DecisionTree<Factor>, pseudoCountSize: Double) : Map<RandomVariable, ProbTree>{
    agentVocab.associate { rv ->
        Pair(rv, emptyProbTree(rv, bestPSets[rv]!!, pseudoCountSize, 0.7))
    }
}
*/

