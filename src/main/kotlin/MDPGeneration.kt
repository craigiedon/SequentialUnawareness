import Utils.random
import Utils.sampleNoReplacement
import Utils.shuffle
import java.util.*

/*
fun generateMDP(numVocab : Int, rewardDomSize : Int, numActions : Int, discount : Double, maxGoalReachability : Int, maxParents : Int) : MDP{
    val vocab = (1..numVocab).map { RandomVariable("X$it", 2) }.toSet()
    val actions = (1..numActions).map{ "A$it" }.toSet()
    val rewardDomain = sampleNoReplacement(vocab, rewardDomSize)

    val rewardTree = generateRewardTree(rewardDomain)
    val terminalDescriptions = extractGoalDescriptions(rewardDomain, rewardTree)
    val goalStates = terminalDescriptions
        .flatMap { partAssgn -> allAssignments(rewardDomain - partAssgn.keys).map { it + partAssgn } }
        .distinct()

    val dbns = actions.associate { Pair(it, generateDBN(vocab, maxGoalReachability, goalStates, maxParents)) }
    return MDP(vocab, rewardTree, actions, dbns, discount, emptyList(), terminalDescriptions)
}

fun generateRewardTree(rewardDomain : List<RandomVariable>) : DecisionTree<Reward> {
    val allZeroStruct : DecisionTree<Reward> = generateDTStruct(rewardDomain, random(0.5, 0.75), 0.0)
    fun fillInSparseRewards(dt : DecisionTree<Reward>) : DecisionTree<Reward>{
        when(dt){
            is DTLeaf -> return if(Math.random() > 0.5){
                DTLeaf(random(1, 100).toDouble())
            } else{
                DTLeaf(0.0)
            }
            is DTDecision -> if(dt.passBranch is DTLeaf && dt.failBranch is DTLeaf){
                if(Math.random() > 0.5){
                    return dt.copy(passBranch = DTLeaf(0.0), failBranch = DTLeaf(random(1, 100).toDouble()))
                }
                else{
                    return dt.copy(passBranch = DTLeaf(random(1, 100).toDouble()), failBranch  = DTLeaf(0.0))
                }
            }
            else{
                return dt.copy(passBranch = fillInSparseRewards(dt.passBranch), failBranch = fillInSparseRewards(dt.failBranch))
            }
        }
    }
    return fillInSparseRewards(allZeroStruct)
}

fun <T> generateDTStruct(domain : List<RandomVariable>, pruneProportion : Double, dummyVal : T) : DecisionTree<T>{
    fun genFullDT(d: List<RandomVariable>) : DecisionTree<T>{
        return if(d.isEmpty()){
            DTLeaf(dummyVal)
        } else{
            val shuffledD = shuffle(d)
            DTDecision(Pair(shuffledD.first(), 0), genFullDT(shuffledD.drop(1)), genFullDT(shuffledD.drop(1)))
        }
    }

    val nodesToPrune : Int = maxOf(domain.size + 1, (numAssignments(domain) * (1.0 - pruneProportion)).toInt())
    val fullDT = genFullDT(domain)
    return cutDTSize(fullDT, nodesToPrune, dummyVal)
}

fun <T> cutDTSize(dt : DecisionTree<T>, nodesToPrune : Int, dummyVal: T) : DecisionTree<T>{
    return when(dt){
        is DTLeaf -> dt
        is DTDecision -> if(nodesToPrune == 0) {
            dt
        }
        else{
            val decisionCounts = countDecisionNodeKeys(dt)
            val removalCandidate = getAllBaseDecisions(dt).filter{ decisionCounts[it.rvTest.first]!! > 1 }.random()
            val updatedDT = replaceNode(dt, removalCandidate, DTLeaf(dummyVal))
            cutDTSize(updatedDT, nodesToPrune - 1, dummyVal)
        }
    }
}

fun <T> countDecisionNodeKeys(dt : DecisionTree<T>) : Map<RandomVariable, Int>{
    val counts = HashMap<RandomVariable, Int>()
    fun recCountDecNodeKeys(node : DecisionTree<T>){
        if(node is DTDecision){
            counts[node.rvTest.first] = (counts[node.rvTest.first] ?: 0) + 1
        }
    }
    recCountDecNodeKeys(dt)
    return counts
}

fun <T> getAllBaseDecisions(dt : DecisionTree<T>) : List<DTDecision<T>>{
    return when(dt){
        is DTLeaf -> emptyList()
        is DTDecision -> if(dt.passBranch is DTLeaf && dt.failBranch is DTLeaf){
            listOf(dt)
        } else{
            getAllBaseDecisions(dt.passBranch) + getAllBaseDecisions(dt.failBranch)
        }
    }
}


fun extractGoalDescriptions(rewardDomain: Collection<RandomVariable>, rewardTreeBinaryVars : DecisionTree<Reward>) : List<RVAssignment>{
    if(rewardDomain.any{ it.domainSize != 2}){
        throw NotImplementedError("Only handles binary values variables for now")
    }
    fun recExtractGDs(dt : DecisionTree<Reward>, partialRVAssignment : RVAssignment) : List<RVAssignment>{
        when(dt){
            is DTLeaf -> {
                return if(dt.value == 0.0){
                    emptyList()
                } else{
                    listOf(partialRVAssignment)
                }
            }
            is DTDecision -> {
                val rv = dt.rvTest.first
                val passAssgn = Pair(rv, dt.rvTest.second)
                val failAssgn = Pair(rv, 1 - dt.rvTest.second)
                return recExtractGDs(dt.passBranch, partialRVAssignment + passAssgn) +
                    recExtractGDs(dt.failBranch, partialRVAssignment + failAssgn)
            }
        }
    }

    return recExtractGDs(rewardTreeBinaryVars, emptyMap())
}

fun <T> replaceNode(dt: DecisionTree<T>, toReplace: DecisionTree<T>, replaceWith: DecisionTree<T>) : DecisionTree<T>{
    return when(dt){
        toReplace -> replaceWith
        is DTDecision -> return dt.copy(passBranch = replaceNode(dt.passBranch, toReplace, replaceWith), failBranch = replaceNode(dt.failBranch, toReplace, replaceWith))
        is DTLeaf -> throw IllegalArgumentException("Cannot find specified node in full DT")
    }
}

fun generateDBN(vocab : Collection<RandomVariable>, maxGoalReachability : Int, goalStates : List<RVAssignment>, maxParents : Int) : DynamicBayesNet{
    // Generate reachability graph for full vocab
    val reachabilityGraph : Map<RVAssignment, RVAssignment> = genReachabilityGraph(vocab, maxGoalReachability, goalStates.toSet())
    // Pick random parents for each variable
    val parentMap = vocab.associate { Pair(it, sampleNoReplacement(vocab, random(1, maxParents + 1))) }
    // Generate CPT structures
    val cptStructs = parentMap.mapValues { generateDTStruct(it.value, 0.5, Factor(listOf(it.key), uniformJoint(listOf(it.key)))) }
    val cpts = cptStructs.mapValues { genCPTWithReachabilityRestrictions(it.value, reachabilityGraph) }
    return cpts
}

fun genReachabilityGraph(vocab : Collection<RandomVariable>, maxGoalReachability: Int, goalStates: Set<RVAssignment>) : Map<RVAssignment, RVAssignment>{
    val possibleAssignments = shuffle(allAssignments(vocab.toList()))
    val partitionSize = possibleAssignments.size / maxGoalReachability
    var targetStates = goalStates
    val reachabilityGroups = possibleAssignments
        .withIndex()
        .groupBy { it.index / partitionSize }
        .map { it.value.map { (i, assgn) -> assgn } }
    val reachabilityMap = HashMap<RVAssignment, RVAssignment>()
    for(sourceStates in reachabilityGroups){
        for(sourceState in sourceStates){
            reachabilityMap[sourceState] = targetStates.random()
        }
        targetStates = sourceStates.toSet()
    }
    return reachabilityMap
}

fun genCPTWithReachabilityRestrictions(dtStruct : DecisionTree<Factor>, reachabilityMap : Map<RVAssignment, RVAssignment>) : DecisionTree<Factor>{
    // For each node in each CPT, look at reachability graph (which should be a hashmap). Are you required to make this one nonzero?
    // If not, theres a chance it will be sparse! (Unless it is the like two leaf child case)
    // You are done! Hopefully!

    // When its an arbitrary decision node, who cares
    // When its a decision node and one of the children is a leaf, also who cares
    // When its a leaf, have to ask what one is required to reach from this state, and set the factors accordingly
    // When its a decision node with two leaves as children, need to ensure that the distributions are not the same.
    when(dtStruct){
        is DTLeaf -> return if(Math.random() > 0.5){
            DTLeaf(random(1, 100).toDouble())
        } else{
            DTLeaf(0.0)
        }
        is DTDecision -> if(dt.passBranch is DTLeaf && dt.failBranch is DTLeaf){
            if(Math.random() > 0.5){
                return dt.copy(passBranch = DTLeaf(0.0), failBranch = DTLeaf(random(1, 100).toDouble()))
            }
            else{
                return dt.copy(passBranch = DTLeaf(random(1, 100).toDouble()), failBranch  = DTLeaf(0.0))
            }
        }
        else{
            return dt.copy(passBranch = fillInSparseRewards(dt.passBranch), failBranch = fillInSparseRewards(dt.failBranch))
        }
    }
}
*/