import Utils.project
import Utils.random

data class Expert(
    val historyWindow: Int,
    val mistakeProportion: Double,
    val adviceInterval : Int,
    val knownAgentVocab : MutableSet<RandomVariable>,
    val problemMDP : MDP,
    val qFuncs: Map<Action, QTree>
){
    val agentOptimalActionHistory = ArrayList<Boolean>()
    val stateHist = HashMap<TimeStamp, RVAssignment>()
    var lastAdvice = 0
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

data class MisunderstandingResolution(val firstAssignment : Pair<RandomVariable, Int>, val secondAssignment : Pair<RandomVariable, Int>)

fun resolveMisunderstanding(t1: TimeStamp, t2: TimeStamp, expert: Expert) : MisunderstandingResolution {
    val state1 = expert.stateHist[t1]!!
    val state2 = expert.stateHist[t2]!!
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