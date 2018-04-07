import Utils.random
import org.apache.commons.math3.util.MathUtils
import java.util.*

data class WeightedAssignment(val assignment : RVAssignment, val weight : Double)

fun generateSamples(scope : List<RandomVariable>, numSamples : Int) =
    (1..numSamples).map { generateSample(scope) }

@JvmName("generateSampleUniform")
fun generateSample(scope : List<RandomVariable>) =
    scope.associate { Pair(it, random(0, it.domainSize)) }

fun generateSamples(bn : BayesNet, numSamples : Int) : List<RVAssignment>{
    val sortedNodes = topologicalSort(bn.nodes.values.toList())
    val samples = ArrayList<RVAssignment>()
    for(i in 0..numSamples-1){
        samples.add(generateSample(sortedNodes))
    }
    return samples
}

fun generateSample(bn : BayesNet) : RVAssignment{
    val sortedNodes = topologicalSort(bn.nodes.values.toList())
    return generateSample(sortedNodes)
}

fun generateSample(sortedNodes : List<BNode>, prevAssignments : RVAssignment = emptyMap()) : RVAssignment{
    val sampleMap = HashMap(prevAssignments)
    for(node : BNode in sortedNodes){
        sampleMap[node.rv] = genValueGivenParents(node, sampleMap)
    }
    return sampleMap
}

fun generateSample(dbn : DynamicBayesNet, prevContext : RVAssignment) =
    dbn.mapValues{(_, cpdTree) ->
        val matchingParam = matchLeaf(cpdTree, prevContext).value
        sample(matchingParam.values)
    }

fun generateSampleWithIntervention(bn : BayesNet, clampedVars : RVAssignment) : RVAssignment {
    val unclampedSortedNodes = topologicalSort(bn.nodes.values.toList()).filter { !clampedVars.keys.contains(it.rv) }

    val sampleMap = HashMap<RandomVariable, Int>(clampedVars)
    for(node in unclampedSortedNodes){
        sampleMap[node.rv] = genValueGivenParents(node, sampleMap)
    }

    return sampleMap
}

fun genValueGivenParents(node : BNode, assignment: Map<RandomVariable, Int>) : Int{
    val familyAssignment = HashMap(assignment.filterKeys { node.parents.contains(it) })
    if(!familyAssignment.keys.containsAll(node.parents)){
        throw IllegalArgumentException("Must provide values for all parents")
    }

    var threshold = 0.0
    val randomVal = Math.random()

    for(i in 0..node.rv.domainSize - 1){
        familyAssignment[node.rv] = i
        val assgnIndex = assignmentToIndex(familyAssignment, listOf(node.rv) + node.parents)
        threshold += node.cpt[assgnIndex]
        if(randomVal < threshold){
            return i
        }
    }

    throw IllegalStateException("CPT values don't appear to sum to one")
}

fun sample(distribution : List<Double>) : Int{
    if(!doubleEquality(distribution.sum(), 1.0)){
        throw IllegalArgumentException("Probabilities must sum to 1")
    }

    val randomVal = Math.random()
    var thresholdTotal = 0.0
    for((i, prob) in distribution.withIndex()){
        thresholdTotal += prob
        if(randomVal < thresholdTotal){
            return i
        }
    }
    throw IllegalStateException("Problem with distribution: Random val: $randomVal, Threshold: $thresholdTotal")
}