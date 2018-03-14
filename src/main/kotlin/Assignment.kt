import java.util.*

typealias  RVAssignment = Map<RandomVariable, Int>

fun allAssignments(rvs : List<RandomVariable>) : List<RVAssignment>{
    val strides = getStrides(rvs)
    return (0..numAssignments(rvs) - 1)
            .map { i -> indexToAssignment(i, strides, rvs).withIndex()
                    .associate { Pair(rvs[it.index], it.value) } }
}


fun numAssignments(rvs : Collection<RandomVariable>) = rvs.fold(1, {acc, rv -> acc * rv.domainSize})
fun lnNumAssignments(rvs : Collection<RandomVariable>) = rvs.fold(0.0, {acc, rv -> acc + Math.log(rv.domainSize.toDouble())})

fun getStrideConverter(originalScope: List<RandomVariable>, convertedScope: List<RandomVariable>) : List<Int>{
    val subScopeStride = getStrides(convertedScope)
    val subStride = IntArray(originalScope.size).toMutableList()

    for (i in originalScope.indices){
        if (convertedScope.contains(originalScope[i])){
            subStride[i] = subScopeStride[convertedScope.indexOf(originalScope[i])]
        }
    }
    return subStride
}

fun indexToAssignment(index : Int, stride : List<Int>, orderedVocab : List<RandomVariable>) : IntArray{
    if(stride.size != orderedVocab.size){
        throw IllegalArgumentException("Different number of stride entries to assignment vocabulary size")
    }
    val assignments = IntArray(stride.size)
    for (i in stride.indices) {
        assignments[i] = Math.floor(index / stride[i].toDouble()).toInt() % orderedVocab[i].domainSize
    }
    return assignments
}

fun indexToAssignment(index : Int, orderedVocab : List<RandomVariable>) =
    indexToAssignment(index, getStrides(orderedVocab), orderedVocab)


fun convertIndex(index : Int, originalVocabOrdered : List<RandomVariable>, newVocab : List<RandomVariable>) : Int{
    val originalAssignment = indexToAssignment(index, getStrides(originalVocabOrdered), originalVocabOrdered)
    val convertedAssignment = newVocab.map {
        if(originalVocabOrdered.contains(it))
            originalAssignment[originalVocabOrdered.indexOf(it)]
        else
            0
    }.toIntArray()
    val convertedIndex = assignmentToIndex(convertedAssignment, getStrides(newVocab))
    return convertedIndex
}

fun assignmentToIndex(assignment : IntArray, stride : List<Int>) =
        assignment.indices.sumBy { assignment[it] * stride[it] }

fun assignmentToIndex(assignment: RVAssignment, scope : List<RandomVariable>) : Int{
    val strides = getStrides(scope)
    return scope.indices.sumBy { assignment[scope[it]]!! * strides[it] }
}

fun getStrides(rvs : List<RandomVariable>) : List<Int>{
    var prevProduct = 1
    val strides = ArrayList<Int>()
    for(i in rvs.indices){
        strides.add(prevProduct)
        prevProduct *= rvs[i].domainSize
    }
    return strides
}

fun hideVars(data : List<RVAssignment>, varsToHide : Collection<RandomVariable>) =
    data.map { trial -> hideVars(trial, varsToHide)}

@JvmName("hideVarsInTrial")
fun hideVars(trial : RVAssignment, varsToHide: Collection<RandomVariable>) =
    trial.filterKeys{!varsToHide.contains(it)}
