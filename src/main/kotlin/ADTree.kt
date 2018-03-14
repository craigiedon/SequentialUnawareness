import java.util.*

class ADTree(var rootNode : ADNode, val buildParams : ADBuildParams, val orderedVocab : List<RandomVariable>)
data class ADBuildParams(val leafListThresh : Int, val upperBuffThresh: Int, val lowerBuffThresh: Int, val mcvRatio : Double, val maxDepth : Int)

sealed class ADNode(var unweightedCount: Int, var weightedCount : Double){
    class SubTree(unweightedCount: Int, weightedCount : Double, val varyNodes : LinkedHashMap<RandomVariable, VaryNode>) : ADNode(unweightedCount, weightedCount){
        override fun toString() : String{
            return "Count: $unweightedCount Vary Nodes: ${varyNodes.keys}"
        }
    }
    class LeafList(val relevantSamples : MutableList<WeightedAssignment>) : ADNode(relevantSamples.size, relevantSamples.sumByDouble { it.weight })
}

sealed class VaryNode{
    class BufferedVaryNode(val leafNodes : List<ADNode.LeafList>) : VaryNode()
    class ExpandedVaryNode(val mostCommonValue: Int, val adNodes : MutableList<ADNode?>) : VaryNode()
}

fun maxDepth(adt : ADTree) = maxDepthAD(adt.rootNode)
fun maxDepthVN(vn : VaryNode) : Int =
        when(vn){
            is VaryNode.BufferedVaryNode -> 0
            is VaryNode.ExpandedVaryNode -> (vn.adNodes.filterNotNull().map(::maxDepthAD).max() ?: 0)
        }
fun maxDepthAD(adn : ADNode) : Int =
        when(adn){
            is ADNode.LeafList -> 0
            is ADNode.SubTree -> 1 + (adn.varyNodes.values.map(::maxDepthVN).max() ?: 0)
        }

fun avgDepth(adt: ADTree) = avgDepthAD(adt.rootNode)
fun avgDepthAD(adn : ADNode) : Double =
        when(adn){
            is ADNode.LeafList -> 0.0
            is ADNode.SubTree -> 1.0 + (adn.varyNodes.values.map(::maxDepthVN )).average()
        }
fun avgDepthVN(vn : VaryNode) : Double =
        when(vn){
            is VaryNode.BufferedVaryNode -> 0.0
            is VaryNode.ExpandedVaryNode -> (vn.adNodes.filterNotNull().map(::maxDepthAD)).average()
        }

@JvmName("buildADTreeUnweighted")
fun buildADTree(vocab : List<RandomVariable>, data : List<RVAssignment>, buildParams: ADBuildParams) =
    buildADTree(vocab, data.map { WeightedAssignment(it, 1.0) }, buildParams)

fun buildADTree(vocab : List<RandomVariable>, data : List<WeightedAssignment>, buildParams: ADBuildParams) =
        ADTree(buildADNode(vocab, 0, data, data.indices.toList(), buildParams, 0), buildParams, vocab)

fun buildADNode(vocab : List<RandomVariable>, currentRVIndex : Int, data : List<WeightedAssignment>, recordNums : List<Int>, buildParams : ADBuildParams, currentDepth: Int) : ADNode{
    if(vocab.isEmpty()){
        throw IllegalArgumentException("Cannot build AD Tree over empty scope")
    }
    val unweightedCount = recordNums.size
    if(unweightedCount <= buildParams.leafListThresh || currentDepth == buildParams.maxDepth){
        return ADNode.LeafList(recordNums.map { data[it] }.toMutableList())
    }
    val weightedCount = recordNums.sumByDouble{data[it].weight}

    val varyNodes = (currentRVIndex..vocab.size-1).associateTo(LinkedHashMap(), { Pair(vocab[it], buildVaryNode(vocab, it, data, recordNums, buildParams, currentDepth)) })
    return ADNode.SubTree(unweightedCount, weightedCount, varyNodes)
}

fun buildVaryNode(vocab : List<RandomVariable>, currentRVIndex : Int, data : List<WeightedAssignment>, recordNums : List<Int>, buildParams : ADBuildParams, currentDepth : Int) : VaryNode{
    val rv = vocab[currentRVIndex]
    val childNums = (0..rv.domainSize - 1).map { ArrayList<Int>()}
    for(recordNum in recordNums){
        val attributeVal = data[recordNum].assignment[rv]!!
        childNums[attributeVal].add(recordNum)
    }

    val mostCommonValue = childNums.map { it.size.toDouble() }.withIndex().maxBy { it.value }!!
    val secondMostCommonValue = childNums.map{ it.size.toDouble() }.withIndex().filter { it.index != mostCommonValue.index }.maxBy { it.value }!!

    if(recordNums.size > buildParams.upperBuffThresh || (recordNums.size > buildParams.lowerBuffThresh && secondMostCommonValue.value / mostCommonValue.value < buildParams.mcvRatio)){
        val childADNodes = ArrayList<ADNode?>()
        for(domValue in 0..rv.domainSize-1){
            if(childNums[domValue].size == 0 || domValue == mostCommonValue.index) {
                childADNodes.add(null)
            }
            else{
                childADNodes.add(buildADNode(vocab, currentRVIndex + 1, data, childNums[domValue], buildParams, currentDepth + 1))
            }

        }
        return VaryNode.ExpandedVaryNode(mostCommonValue.index, childADNodes)
    }
    else{
        return VaryNode.BufferedVaryNode(childNums.map { indexes -> ADNode.LeafList(indexes.map { data[it] }.toMutableList()) })
    }
}

fun makeWeightedCountTable(queryVocab : List<RandomVariable>, adTree : ADTree) : CountTable{
    val root = adTree.rootNode
    if(queryVocab.size > adTree.buildParams.maxDepth){
        throw IllegalArgumentException("Cannot have query size bigger than max depth of AD Tree")
    }
    if(queryVocab.any { !adTree.orderedVocab.contains(it) }){
        throw IllegalArgumentException("Querying for rv that does not exist in AD Tree")
    }

    return when(root){
        is ADNode.LeafList -> CountTable(queryVocab.reversed(), samplesToCounts(queryVocab.reversed(), root.relevantSamples))
        is ADNode.SubTree -> {
            val orderedVocab : List<RandomVariable> = root.varyNodes.map{it.key}.filter { queryVocab.contains(it)}
            makeCTRec(orderedVocab, root)
        }
    }
}

fun makeCTRec(orderedVocab : List<RandomVariable>, adNode: ADNode) : CountTable{
    if(orderedVocab.isEmpty()){
        val counts = DoubleArray(1)
        counts[0] = adNode.weightedCount
        return CountTable(emptyList(), counts)
    }

    val queryVar = orderedVocab[0]

    return when(adNode) {
        is ADNode.LeafList -> CountTable(orderedVocab.reversed(), samplesToCounts(orderedVocab.reversed(), adNode.relevantSamples))
        is ADNode.SubTree -> {
            val varyNode = adNode.varyNodes[queryVar]!!
            val countArrays : Array<DoubleArray?> = when(varyNode){
                is VaryNode.BufferedVaryNode -> varyNode.leafNodes.map { makeCTRec(orderedVocab.drop(1), it).counts }.toTypedArray()
                is VaryNode.ExpandedVaryNode -> makeCTRec(orderedVocab, varyNode, adNode)
            }
            CountTable(orderedVocab.reversed(), combineCounts(countArrays))
        }
    }

}

fun makeCTRec(orderedVocab : List<RandomVariable>, varyNode : VaryNode.ExpandedVaryNode, parentADNode : ADNode) : Array<DoubleArray?>{
    val queryVar = orderedVocab[0]
    val countArrays: Array<DoubleArray?> = arrayOfNulls(queryVar.domainSize)

    for (domainVal in 0..queryVar.domainSize - 1) {
        if (domainVal != varyNode.mostCommonValue) {
            val nextADNode = varyNode.adNodes[domainVal]
            if (nextADNode == null) {
                countArrays[domainVal] = DoubleArray(numAssignments(orderedVocab.drop(1)))
            } else {
                countArrays[domainVal] = makeCTRec(orderedVocab.drop(1), nextADNode).counts
            }
        }
    }

    countArrays[varyNode.mostCommonValue] = subCounts(makeCTRec(orderedVocab.drop(1), parentADNode).counts, sumCounts(countArrays.filterNotNull()))
    if(countArrays[varyNode.mostCommonValue]!!.any{it < 0}){
        println("Can't have a negative state!")
    }
    return countArrays
}

fun combineCounts(countArrays : Array<DoubleArray?>) : DoubleArray{
    val combinedCountsSize = countArrays.map { it!!.size }.sum()
    val combinedCounts = DoubleArray(combinedCountsSize)
    var i = 0
    for (countArray in countArrays) {
        for (count in countArray!!) {
            combinedCounts[i] = count
            i++
        }
    }
    return combinedCounts
}

fun subCounts(count1 : DoubleArray, count2 : DoubleArray) : DoubleArray{
    val result = DoubleArray(count1.size)
    for(i in count1.indices){
        result[i] = maxOf(0.0, count1[i] - count2[i]) // Should never really be less than zero, but sometimes tips under due to rounding errors
    }
    return result
}

fun sumCounts(listOfCounts : List<DoubleArray>) : DoubleArray {
    val sumArray = DoubleArray(listOfCounts[0].size)
    for(counts in listOfCounts){
        for(i in counts.indices){
            sumArray[i] += counts[i]
        }
    }
    return sumArray
}


@JvmName("UpdateADTUnweighted")
fun updateADTree(adTree : ADTree, additionalData : List<RVAssignment>) =
    updateADTree(adTree, additionalData.map { WeightedAssignment(it, 1.0) })

fun updateADTree(adTree : ADTree, additionalData : List<WeightedAssignment>){
    adTree.rootNode = updateADNode(adTree.orderedVocab, 0, adTree.rootNode, additionalData, additionalData.indices.toList(), adTree.buildParams, 0)
}

fun updateADNode(vocab : List<RandomVariable>, currentRVIndex : Int, adNode : ADNode, additionalData: List<WeightedAssignment>, recordNums : List<Int>, buildParams : ADBuildParams, currentDepth : Int) : ADNode{
    val newSamples = recordNums.map { additionalData[it] }
    adNode.unweightedCount += newSamples.size
    adNode.weightedCount += newSamples.sumByDouble { it.weight }
    when(adNode){
        is ADNode.LeafList -> {
            adNode.relevantSamples.addAll(newSamples)
            if(adNode.unweightedCount > buildParams.leafListThresh && currentDepth < buildParams.maxDepth){
                return buildADNode(vocab, currentRVIndex, adNode.relevantSamples, adNode.relevantSamples.indices.toList(), buildParams, currentDepth)
            }
        }
        is ADNode.SubTree ->{
            for(i in currentRVIndex..vocab.size-1){
                val oldVaryNode = adNode.varyNodes[vocab[i]]!!
                adNode.varyNodes[vocab[i]] = updateVaryNode(vocab, i, oldVaryNode, additionalData, recordNums, buildParams, currentDepth)
            }
        }
    }
    return adNode
}

fun updateVaryNode(vocab: List<RandomVariable>, currentRVIndex: Int, varyNode: VaryNode, additionalData: List<WeightedAssignment>, recordNums: List<Int>, buildParams: ADBuildParams, currentDepth : Int) : VaryNode{
    val rv = vocab[currentRVIndex]
    val childNums = (0..rv.domainSize - 1).map{ArrayList<Int>()}
    for(recordNum in recordNums){
        val attributeVal = additionalData[recordNum].assignment[rv]!!
        childNums[attributeVal].add(recordNum)
    }

    when(varyNode){
        is VaryNode.ExpandedVaryNode -> {
            for(i in varyNode.adNodes.indices){
                val oldAD = varyNode.adNodes[i]
                if(i != varyNode.mostCommonValue && oldAD != null){
                    varyNode.adNodes[i] = updateADNode(vocab, currentRVIndex + 1, oldAD, additionalData, childNums[i], buildParams, currentDepth + 1)
                }
                else if(i != varyNode.mostCommonValue && childNums[i].isNotEmpty()){
                    val relevantSamples = childNums[i].map { additionalData[it] }
                    varyNode.adNodes[i] = buildADNode(vocab, currentRVIndex + 1, relevantSamples, relevantSamples.indices.toList(), buildParams, currentDepth + 1)
                }
            }
        }
        is VaryNode.BufferedVaryNode -> {
            for(i in varyNode.leafNodes.indices){
                val newRelevantSamples = childNums[i].map{additionalData[it]}
                varyNode.leafNodes[i].relevantSamples.addAll(childNums[i].map { additionalData[it] })
                varyNode.leafNodes[i].weightedCount += newRelevantSamples.sumByDouble { it.weight }
                varyNode.leafNodes[i].unweightedCount += newRelevantSamples.size
            }

            val counts = varyNode.leafNodes.map { it.unweightedCount }
            val totalCount = counts.sum()
            val mcv = counts.withIndex().maxBy { it.value }!!
            val secondMcv = counts.withIndex().filter { it != mcv }.maxBy { it.value }!!
            if(totalCount > buildParams.upperBuffThresh || (totalCount > buildParams.lowerBuffThresh && secondMcv.value.toDouble() / mcv.value.toDouble() < buildParams.mcvRatio)){
                val newADNodes = varyNode.leafNodes.mapIndexed { index, leafList ->
                    if(index == mcv.index || leafList.unweightedCount == 0){
                        null
                    }
                    else{
                        buildADNode(vocab, currentRVIndex + 1, leafList.relevantSamples, leafList.relevantSamples.indices.toList(), buildParams, currentDepth + 1)
                    }
                }
                return VaryNode.ExpandedVaryNode(mcv.index, newADNodes.toMutableList())
            }
        }
    }

    return varyNode
}

fun multiplyADTCounts(multFactor : Double, adt : ADNode){
    adt.weightedCount *= multFactor
    when(adt){
        is ADNode.LeafList -> {
            for(i in adt.relevantSamples.indices){
                val oldSample = adt.relevantSamples[i]
                adt.relevantSamples[i] = WeightedAssignment(oldSample.assignment, oldSample.weight * multFactor)
            }
        }
        is ADNode.SubTree -> {
            for((_,varyNode) in adt.varyNodes){
                multiplyVaryNode(multFactor, varyNode)
            }
        }
    }
}

fun multiplyVaryNode(multFactor : Double, varyNode :VaryNode){
    when(varyNode){
        is VaryNode.BufferedVaryNode -> {
            for(leafNode in varyNode.leafNodes){
                multiplyADTCounts(multFactor, leafNode)
            }
        }
        is VaryNode.ExpandedVaryNode -> {
            for(adNode in varyNode.adNodes.filterNotNull()){
                multiplyADTCounts(multFactor, adNode)
            }
        }
    }
}
