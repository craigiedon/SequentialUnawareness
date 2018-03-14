package Utils

import cc.mallet.types.Dirichlet
import java.util.*
import BayesNet
import RandomVariable
import BNode
import numAssignments

fun <T> shuffle(l : List<T>) : List<T>{
    val gen = Random()
    val resultingList = ArrayList(l)
    for (n in (resultingList.size - 1) downTo 1){
        val randomIndex = gen.nextInt(n + 1)
        val tempVal = resultingList[n]
        resultingList[n] = resultingList[randomIndex]
        resultingList[randomIndex] = tempVal
    }

    return resultingList
}

fun <T> Set<T>.random() : T =
    this.toList().random()

fun <T> List<T>.random() : T{
    if(this.isEmpty()){
        throw IllegalArgumentException("Cannot choose random value from empty list")
    }
    return this[(Math.random() * this.size).toInt()]
}

fun randomList(lowerBound : Double, upperBound : Double, length : Int) =
    (1..length).map { random(lowerBound, upperBound) }

fun random(lowerBound : Double, upperBound : Double) : Double {
    if(lowerBound > upperBound) throw IllegalArgumentException("Lower Bound must be less than Upper Bound")
    return Math.random() * (upperBound - lowerBound) + lowerBound
}

fun random(lbInclusive : Int, ubExclusive : Int) : Int {
    if(lbInclusive > ubExclusive) throw IllegalArgumentException("Lower Bound must be less than Upper Bound")
    return (Math.random() * (ubExclusive - lbInclusive)).toInt() + lbInclusive
}


fun <T> sampleNoReplacement(vals : List<T>, numSamples : Int) : List<T>{
    if(vals.size < numSamples){
        throw IllegalArgumentException("Samples asked for exceeds size of list")
    }

    return shuffle(vals).take(numSamples)
}

fun randomParams(parentStructure : Map<RandomVariable, Set<RandomVariable>>) : BayesNet {
    val nodes = HashMap<RandomVariable, BNode>()
    for((rv, parents) in parentStructure){
        val orderedParents = parents.toList()
        val cpt = randomCPT(rv, orderedParents)
        nodes[rv] = BNode(rv, orderedParents, cpt)
    }
    return BayesNet(nodes)
}

fun randomCPT(child : RandomVariable, parents : List<RandomVariable>) : List<Double>{
    val cptVals = ArrayList<Double>()
    val uniformMultiDim = Dirichlet(child.domainSize)
    for (i in 1..numAssignments(parents)){
        val conditionalDist = uniformMultiDim.nextDistribution().toList()
        cptVals.addAll(conditionalDist)
    }

    return cptVals
}
