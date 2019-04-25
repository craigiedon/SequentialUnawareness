typealias Factor = RVTable<Double>

fun extractFactors(bn : BayesNet) : List<Factor> {
    return bn.nodes.map { Factor(listOf(it.key) + it.value.parents, it.value.cpt)  }
}


fun normalize(factor: Factor) : Factor{
    val totalMass = factor.values.sum()
    return Factor(factor.scope, factor.values.map { it / totalMass })
}

fun sumOut(factor : Factor, sumOutVars : List<RandomVariable>) : Factor{
    val fullVars = factor.scope
    val fullStride = getStrides(fullVars)

    val reducedVars = factor.scope.filter { !sumOutVars.contains(it) }
    val reducedSubsetIndices = reducedVars.map { fullVars.indexOf(it) }
    val reducedVarProbs = DoubleArray(numAssignments(reducedVars))
    val reducedStride = getStrides(reducedVars)

    for (i in factor.values.indices){
        val fullAssignment = indexToAssignment(i, fullStride, fullVars)
        val reducedAssignment = reducedSubsetIndices.map { fullAssignment[it] }.toIntArray()
        val reducedIndex = assignmentToIndex(reducedAssignment, reducedStride)
        reducedVarProbs[reducedIndex] += factor.values[i]
    }

    return Factor(reducedVars, reducedVarProbs.toList())
}

fun scale(f1 : Factor, scale : Double) : Factor =
    f1.copy(values = f1.values.map { it * scale })


fun product(f1 : Factor, f2 : Factor) : Factor{
    /* I think the reason it is ok to ignore f1 here is that
       If it doesn't correspond to a particular scope, it is essentially
       a constant value, which will usually get normalized out at the end anyways
     */
    if(f1.scope.isEmpty()){
        if(f1.values.size != 1){
            throw IllegalStateException("Factor scope is empty, but conditionalProb table doesn't contain 1 entry. Something is wrong")
        }
        return Factor(f2.scope, f2.values.map { it * f1.values[0] })
    }
    if(f2.scope.isEmpty()){
        if(f2.values.size != 1){
            throw IllegalStateException("Factor scope is empty, but conditionalProb table doesn't contain 1 entry. Something is wrong")
        }
        return Factor(f1.scope, f1.values.map { it * f2.values[0] })
    }

    val scopeUnion = (f1.scope.toSet() union f2.scope.toSet()).toList()
    val fProdResults = DoubleArray(numAssignments(scopeUnion))
    val fullStride = getStrides(scopeUnion)

    val f1Stride = getStrideConverter(scopeUnion, f1.scope)
    val f2Stride = getStrideConverter(scopeUnion, f2.scope)

    for(f_prod_index in fProdResults.indices){
        val assignment = indexToAssignment(f_prod_index, fullStride, scopeUnion)
        val f1Index = assignmentToIndex(assignment, f1Stride)
        val f2Index = assignmentToIndex(assignment, f2Stride)
        try{
            fProdResults[f_prod_index] = f1.values[f1Index] * f2.values[f2Index]
        }
        catch (e : Exception){
            throw e
        }
    }

    return Factor(scopeUnion, fProdResults.toList())
}

fun divide(f1 : Factor, f2 : Factor) : Factor{
    if(f1.scope.isEmpty()){
        return f2
    }
    if(f2.scope.isEmpty()){
        return f1
    }
    val scopeUnion = (f1.scope.toSet() union f2.scope.toSet()).toList()
    val fProdResults = DoubleArray(numAssignments(scopeUnion))
    val fullStride = getStrides(scopeUnion)

    val f1Stride = getStrideConverter(scopeUnion, f1.scope)
    val f2Stride = getStrideConverter(scopeUnion, f2.scope)

    for(f_prod_index in fProdResults.indices){
        val assignment = indexToAssignment(f_prod_index, fullStride, scopeUnion)
        val f1Index = assignmentToIndex(assignment, f1Stride)
        val f2Index = assignmentToIndex(assignment, f2Stride)
        fProdResults[f_prod_index] = if (f1.values[f1Index] == 0.0 && f2.values[f2Index] == 0.0) 0.0 else f1.values[f1Index] / f2.values[f2Index]
    }

    return Factor(scopeUnion, fProdResults.toList())
}
