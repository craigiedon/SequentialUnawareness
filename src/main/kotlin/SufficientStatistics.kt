import java.util.*

typealias SufficientStats = Map<Set<RandomVariable>, CountTable>

class SequentialCountTable(val prevScope : List<RandomVariable>, val nextScope : List<RandomVariable>, val counts : DoubleArray){
    constructor(prevScope: List<RandomVariable>, nextScope: List<RandomVariable>) : this(prevScope, nextScope, DoubleArray(numAssignments(prevScope + nextScope)))
    constructor(prevScope: List<RandomVariable>, nextScope: List<RandomVariable>, seqTrials : List<SequentialTrial>) : this(prevScope, nextScope){
        updateCounts(seqTrials)
    }

    fun updateCounts(seqTrials : List<SequentialTrial>) {
        seqTrials.forEach{ updateCounts(it) }
    }

    fun updateCounts(seqTrial : SequentialTrial){
        counts[getIndex(seqTrial.prevState, seqTrial.currentState)] += 1.0
    }

    fun getCount(prevAssignment : RVAssignment, nextAssignment : RVAssignment) : Double{
        return counts[getIndex(prevAssignment, nextAssignment)]
    }

    fun getMarginalCount(prevAssignment: RVAssignment) : Double =
        allAssignments(nextScope).sumByDouble { getCount(prevAssignment, it) }

    fun getConditionalCounts(prevAssignment: RVAssignment) : List<Double> {
        val startIndex = getIndex(prevAssignment, nextScope.associate { Pair(it, 0) })
        val endIndex = getIndex(prevAssignment, nextScope.associate { Pair(it, it.domainSize - 1) })
        return counts.slice(startIndex..endIndex)
    }

    private fun getIndex(prevAssignment: RVAssignment, nextAssignment: RVAssignment) : Int{
        val nextAssgnIndex = assignmentToIndex(nextAssignment, nextScope)
        val prevAssgnIndex = assignmentToIndex(prevAssignment, prevScope)
        return nextAssgnIndex + numAssignments(nextScope) * prevAssgnIndex
    }

    override fun toString(): String = "Prev: $prevScope, Next: $nextScope, ${counts.toList()}"
}


class CountTable(val scope : List<RandomVariable>, val counts : DoubleArray){
    constructor(scope : List<RandomVariable>) : this(scope, DoubleArray(numAssignments(scope)))
    constructor(scope : List<RandomVariable>, trials : List<RVAssignment>) : this(scope){
        updateCounts(trials)
    }

    fun updateCounts(assignment: RVAssignment, weight : Double = 1.0){
        val assgnIndex = assignmentToIndex(assignment, this.scope)
        counts[assgnIndex] += weight
    }
    fun updateCounts(samples : List<RVAssignment>){
        samples.forEach { updateCounts(it) }
    }

    fun getCount(assignment : RVAssignment) =
        counts[assignmentToIndex(assignment, scope)]

    override fun toString(): String = scope.toString() + counts.toList()
}

fun initializeStats(desiredStats : List<Set<RandomVariable>>) : SufficientStats{
    return desiredStats.associate { rvFamily -> Pair(rvFamily, CountTable(rvFamily.toList())) }
}

fun scaleCountTable(ct : CountTable, scaleAmount : Double) =
    CountTable(ct.scope, ct.counts.map { it * scaleAmount }.toDoubleArray())
fun scaleStats(stats : SufficientStats, scaleAmount : Double) =
    stats.mapValues { scaleCountTable(it.value, scaleAmount)}

fun projectCounts(countTable : CountTable, projectionVars : List<RandomVariable>) : CountTable{
    if(!countTable.scope.containsAll(projectionVars)){
        throw IllegalArgumentException("Projection Impossible, projection variables not subset of original scope")
    }

    val extraVars = HashSet(countTable.scope) - HashSet(projectionVars)

    val projectedCounts = DoubleArray(numAssignments(projectionVars))
    for(projectedAssignment in allAssignments(projectionVars)){
        val projectedAssignmentIndex = assignmentToIndex(projectedAssignment, projectionVars)
        val summedValue = allAssignments(extraVars.toList())
                .map { exAssignment -> projectedAssignment + exAssignment }
                .sumByDouble { countTable.counts[assignmentToIndex(it, countTable.scope)] }
        projectedCounts[projectedAssignmentIndex] = summedValue
    }

    return CountTable(projectionVars, projectedCounts)
}

@JvmName("SamplesToCountsUnweighted")
fun samplesToCounts(scope : List<RandomVariable>, data : List<RVAssignment>) =
    samplesToCounts(scope, data.map { WeightedAssignment(it, 1.0) })

fun samplesToCounts(scope : List<RandomVariable>, data : List<WeightedAssignment>) : DoubleArray{
    val familyCounts = DoubleArray(numAssignments(scope))
    for((assignment, weight) in data){
        val assgnIndex = assignmentToIndex(assignment,scope)
        familyCounts[assgnIndex] += weight
    }
    return familyCounts
}

fun sufficientStats(structure : BNStruct, adTree : ADTree, cachedStats: SufficientStats = emptyMap()) =
    sufficientStats(listOf(structure), adTree, cachedStats)

@JvmName("sufficientStatsUnweighted")
fun sufficientStats(desiredStats: Collection<Set<RandomVariable>>, data : List<RVAssignment>, cachedStats: SufficientStats = emptyMap()) =
    sufficientStats(desiredStats, data.map { WeightedAssignment(it, 1.0) }, cachedStats)

fun sufficientStats(desiredStats : Collection<Set<RandomVariable>>, data : List<WeightedAssignment>, cachedStats : SufficientStats = emptyMap()) : SufficientStats{
    return desiredStats.associate { family ->
        if(cachedStats.containsKey(family)){
            Pair(family, cachedStats[family]!!)
        }
        else{
            val orderedFamily = family.toList()
            Pair(family, CountTable(orderedFamily, samplesToCounts(orderedFamily, data)))
        }
    }
}

fun sufficientStats(structures : List<BNStruct>, adTree : ADTree, cachedStats : Map<Set<RandomVariable>, CountTable> = emptyMap()) : SufficientStats {
    val newStats = HashMap<Set<RandomVariable>, CountTable>()
    for(structure in structures){
        for((child, parents) in structure){
            val family = parents + child
            if(cachedStats.containsKey(family)){
                newStats[family] = cachedStats[family]!!
            }
            else{
                newStats[family] = makeWeightedCountTable(family.toList(), adTree)
            }
        }
    }
    if(newStats.values.any { it.counts.any{it.isInfinite() || it.isNaN()} }){
        println("Probable infinity alert")
    }
    return newStats
}
