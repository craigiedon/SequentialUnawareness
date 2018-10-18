import Utils.project
import Utils.random
data class Expert(
    val historyWindow: Int,
    val policyErrorTolerance: Double,
    val adviceInterval : Int,
    val maxEpisodeLength : Int,
    val knownAgentVocab : MutableSet<RandomVariable>,
    val problemMDP : MDP,
    val qFuncs: Map<Action, QTree>
){
    val agentOptimalActionHistory = ArrayList<Boolean>()
    val episodeHistory : MutableList<MutableList<SequentialTrial>> = mutableListOf(ArrayList())
    var lastAdvice = 0
    fun globalHistory(t : TimeStamp) : SequentialTrial{
        val (epNum, localTime) = globalToLocal(episodeHistory, t)
        return episodeHistory[epNum][localTime]
    }

}

fun <T> globalToLocal(localHistories : List<List<T>>, t : TimeStamp) : Pair<Int, Int>{
    var runningTotal = 0
    for((epNum, episode) in localHistories.withIndex()){
        if(runningTotal + episode.size > t){
            val localTime = t - runningTotal
            return Pair(epNum, localTime)
        }
        runningTotal += episode.size
    }
    throw IllegalArgumentException("Time stamp $t is greater than all current episodes")
}

fun relevantEps(epHistory : List<List<SequentialTrial>>, fromGlobalTime : TimeStamp) : List<List<SequentialTrial>> {
    val untilEpsiode = globalToLocal(epHistory, fromGlobalTime).first
    return epHistory.drop(untilEpsiode)
}

fun whatsInRewardScope(knownScope : Set<RandomVariable>, expert : Expert) : Set<RandomVariable>{
    // Give them all the knownVocab extra scope, but also add in one extra from the unknown vocab if you want
    val extraScope = expert.problemMDP.rewardScope - knownScope
    if (extraScope.isEmpty()){
        throw IllegalArgumentException("Agent is asking for reward scope when it knows full scope")
    }

    val (inVocab, outVocab) = extraScope.partition { it in expert.knownAgentVocab }

    val scopeAdvice = (if(outVocab.isNotEmpty()) inVocab + outVocab.random() else inVocab).toSet()
    expert.knownAgentVocab.addAll(scopeAdvice)

    return scopeAdvice
}

/* Note: At the moment, the expert does not need to keep track of which actions the agent knows ---
    All actions are atomic, and agent can express 1UV, so there is no danger of explaining too much
*/
fun betterActionAdvice(prevState : RVAssignment, expert : Expert) : Action{
    return expert.qFuncs.maxBy { (_, qTree) -> matchLeaf(qTree, prevState).value }!!.key
}

fun wasActionOptimal(state : RVAssignment, action : Action, qFuncs: Map<Action, QTree>) : Boolean {
    val values = qFuncs.mapValues { (_, qFunc) -> matchLeaf(qFunc, state).value }
    val bestAction = values.maxBy { it.value }!!.key
    // Question: What if there is more than one best action? Surely you shoulnd't penalize agent for picking a different maximal action from you
    return action == bestAction
}

fun poorRecentPerformance(optimalActionHistory : List<Boolean>, horizon: Int, mistakePropThresh: Double) : Boolean{
    if(mistakePropThresh < 0.0 || mistakePropThresh > 1.0){
        throw IllegalArgumentException("Mistake Proportion must be between 0 and 1")
    }
    if(horizon <= 0){
        throw IllegalArgumentException("Horizon must be positive integer")
    }

    val relevantHistory = optimalActionHistory.takeLast(horizon)
    val mistakeProportion = relevantHistory.count{ optActionTaken -> !optActionTaken } / relevantHistory.size.toDouble()
    return mistakeProportion > mistakePropThresh
}

fun policyErrorEstimate(episodeHistory: List<List<SequentialTrial>>, discountFactor : Double, trueMDP : MDP, trueValues : DecisionTree<Double>) : Double{
    if(episodeHistory.isEmpty())
        return 0.0

    val finishedReturns = episodeHistory
        .dropLast(1)
        .sumByDouble { discountedReturn(it, discountFactor)}

    val lastEp = episodeHistory.last() // Question : What if episode history is empty?
    if(lastEp.isEmpty())
        return 0.0

    val partialReturn = discountedReturn(lastEp, discountFactor)

    // Question : What if final episode has no trials?
    val lastEstimatedReturn = partialReturn + matchLeaf(trueValues, lastEp.last().currentState).value * Math.pow(discountFactor, lastEp.size.toDouble())

    val policyQualityEstimate = (finishedReturns + lastEstimatedReturn) / episodeHistory.size

    val startStates = allAssignments(trueMDP.vocab.toList())
        .asSequence()
        .filter { trueMDP.startStateDescriptions.isEmpty() || trueMDP.startStateDescriptions.any { startDesc -> partialMatch(startDesc, it) } }

    val numStartStates = startStates.count()

    val truePolicyQuality = startStates
        .sumByDouble { startingState ->
            (1.0 / numStartStates) * matchLeaf(trueValues, startingState).value
        }

    return maxOf(0.0, truePolicyQuality - policyQualityEstimate)
}

fun discountedReturn(rewards : List<SequentialTrial>, discount : Double) : Double =
    rewards
        .asSequence()
        .withIndex()
        .sumByDouble { (i, trial) -> Math.pow(discount, i.toDouble()) * trial.reward }

data class MisunderstandingResolution(val firstAssignment : Pair<RandomVariable, Int>, val secondAssignment : Pair<RandomVariable, Int>)

fun resolveMisunderstanding(t1: TimeStamp, t2: TimeStamp, expert: Expert) : MisunderstandingResolution {
    val state1 = expert.globalHistory(t1).prevState
    val state2 = expert.globalHistory(t2).prevState
    if(project(state1, expert.knownAgentVocab) != project(state2, expert.knownAgentVocab)){
        throw IllegalArgumentException("Agent is asking about conflict between two trials which it should see as different")
    }
    if(state1 == state2){
        throw IllegalArgumentException("Agent has observed conflict between two identical states?")
    }

    // Then, pick a variable not in known agent vocab such that the assignment value is different in each of the two stateHist
    val possibleAnswers = state1.keys.filter { rv -> state1[rv]!! != state2[rv]!! }
    val newRV = possibleAnswers.random()
    expert.knownAgentVocab.add(newRV)
    return MisunderstandingResolution(Pair(newRV, state1[newRV]!!), Pair(newRV, state2[newRV]!!))
}

// Reveals a variable which this variable is a parent of (maybe it should also reveal if it is in the reward domain?)
/*
fun whatEffect(rv : RandomVariable, expert: Expert) : Pair<Action, RandomVariable> {
    for((action, dbn) in expert.problemMDP.dbnStructs){
        val rvChildren : PSet = parentMapToChildMap(dbn)[rv]!!
        if(rvChildren.isNotEmpty()){
            val (inVocabChildren, outVocabChildren) = rvChildren.partition { it in expert.knownAgentVocab }
            if(outVocabChildren.isNotEmpty()){
                val child = outVocabChildren.random()
                expert.knownAgentVocab.add(child)
                return Pair(action, child)
            }
            else{
                return Pair(action, inVocabChildren.random())
            }
        }
    }

    throw NotImplementedError("If rv has no children in any dbn, it might be a reward scope var. Implementation does not yet account for this possibility")
}
*/