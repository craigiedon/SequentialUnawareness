import java.util.*

fun main(args : Array<String>){}

fun filterFactor(factor : Factor, lockedAssignment: RVAssignment) : Factor{
    val lockInVars = factor.scope.filter { lockedAssignment.containsKey(it) }
    if(lockInVars.isEmpty()){
        return factor
    }

    val strideMap = factor.scope.zip(getStrides(factor.scope)).associate { it }
    val lockInStride = lockInVars.sumBy { lockedAssignment[it]!! * strideMap[it]!! }
    val remainingVars = factor.scope - lockInVars

    val newProbTable = allAssignments(remainingVars).map{ assignment ->
        val relevantIndex = assignment.entries.sumBy { (rv, rvVal) -> strideMap[rv]!! * rvVal } + lockInStride
        factor.values[relevantIndex]
    }

    return Factor(remainingVars, newProbTable)
}

/*
fun simpleLnQuery(queryAssignment: Map<RandomVariable, Int>, bn : BayesNet) : Double{
    val missingVars = (bn.vocab - queryAssignment.keys).toList()
    val lnProbs = allAssignments(missingVars)
            .map { simpleLnQueryFull(queryAssignment + it, bn) }

    val finiteLnProbs = lnProbs.filter(Double::isFinite)
    if (finiteLnProbs.isEmpty()){
        // Impossible state
        return Math.log(0.0)
    }
    return sumInnerLnProbs(finiteLnProbs)
}

fun simpleLnQueryFull(queryAssignment : Map<RandomVariable, Int>, bn : BayesNet) : Double{
    if (queryAssignment.keys != bn.nodes.keys){
        throw IllegalArgumentException("Full assignment not given - does not cover all variables of BN")
    }

    var logTotalProb = 0.0
    for((rv,node) in bn.nodes){
        logTotalProb += Math.log(node.cpt[assignmentToIndex(queryAssignment, listOf(rv) + node.parents)])
        if(logTotalProb.isNaN()){
            println("Where is this NaN coming from?")
        }
    }
    return logTotalProb
}
*/

fun variableElimination(factors : List<Factor>, varsToElim : List<RandomVariable>) : Factor{
    if(factors.size == 1)
        return sumOut(factors[0], varsToElim)

    val remainingFactors = ArrayList(factors)
    val remainingVarsToElim = ArrayList(varsToElim)
    while(remainingVarsToElim.isNotEmpty()){
        val minDegreeVar = minResultingDegreeVar(remainingVarsToElim, remainingFactors)
        remainingVarsToElim.remove(minDegreeVar)

        val relevantFactors = remainingFactors.filter { it.scope.contains(minDegreeVar) }
        if(relevantFactors.isNotEmpty()){
            remainingFactors.removeAll(relevantFactors)
            val reducedFactor = sumOut(relevantFactors.reduce(::product), listOf(minDegreeVar))
            remainingFactors.add(reducedFactor)
        }
    }

    return remainingFactors.reduce(::product)
}

fun minResultingDegreeVar(rvs : List<RandomVariable>, factors : List<Factor>) : RandomVariable{
    val factorScopes = factors.map { it.scope.toSet() }
    return rvs.minBy { rv ->
        val combinedScope = factorScopes.filter { it.contains(rv) }.flatten().distinct()
        numAssignments(combinedScope)
    }!!
}

fun variableEliminationQuery(queryVars: List<RandomVariable>, evidenceVars : List<RandomVariable>, factors: List<Factor>) : Factor{
    val jointFactor = variableEliminationQuery(queryVars, emptyMap(), factors)
    val evidenceFactor = variableEliminationQuery(evidenceVars, emptyMap(), factors)
    return divide(jointFactor, evidenceFactor)
}

fun variableEliminationQuery(queryVars: List<RandomVariable>, evidence: RVAssignment, bn: BayesNet) =
    variableEliminationQuery(queryVars, evidence, extractFactors(bn))

/*
fun variableEliminationQuery(queryVars: List<RandomVariable>, evidence: RVAssignment, dn: DecisionNetwork) =
    variableEliminationQuery(queryVars, evidence, extractFactors(dn))
*/

fun variableEliminationQuery(queryVars: List<RandomVariable>, evidence: RVAssignment, factors: List<Factor>) : Factor{
    if (queryVars.isEmpty()){
        return Factor(emptyList(), listOf(1.0))
    }

    val eliminationVars = factors
        .flatMap { it.scope }
        .asSequence()
        .distinct()
        .filter { !queryVars.contains(it) }
        .toList()

    val prunedFactors = factors
        .asSequence()
        .map { evidenceApplicationPrune(evidence, it, queryVars) }
        .map { applyEvidence(evidence, it)}
        .toList()

    val unorderedFactor = normalize(variableElimination(prunedFactors, eliminationVars))

    return reorderFactor(unorderedFactor, queryVars)
}

fun variableEliminationQuery(queryAssignment : RVAssignment, bn : BayesNet) : Double{
    if (queryAssignment.isEmpty()){
        throw IllegalArgumentException("Empty query assignment")
    }

    val eliminationVars = bn.vocab.filter { !queryAssignment.keys.contains(it) }
    val filteredFactors = extractFactors(bn).map { filterFactor(it, queryAssignment) }

    val finalFactor = variableElimination(filteredFactors, eliminationVars)
    if(finalFactor.values.size != 1){
        throw IllegalStateException("Final factor should contain only a single value: The assignment probability")
    }

    return finalFactor.values[0]
}

fun reorderFactor(factor : Factor, newArrangement : List<RandomVariable>) : Factor{
    if(factor.scope.toSet() != newArrangement.toSet())
        throw IllegalArgumentException("New scope must be rearrangement of the old one")
    val reorderedProbTable = factor.values.indices.map { factor.values[convertIndex(it, newArrangement, factor.scope)] }
    return Factor(newArrangement, reorderedProbTable)
}

fun evidenceApplicationPrune(evidence: Map<RandomVariable, Int>, fullFactor: Factor, queryVars : List<RandomVariable>) : Factor{
    val relevantEvidence = evidence.filterKeys { fullFactor.scope.contains(it) && !queryVars.contains(it)}
    val reducedScope : List<RandomVariable> = fullFactor.scope - relevantEvidence.keys
    val reducedProbTable = DoubleArray(numAssignments(reducedScope))

    val reducedScopeStride = getStrides(reducedScope)

    for(i in 0 until numAssignments(reducedScope)){
        val newScopeAssignment = indexToAssignment(i, reducedScopeStride, reducedScope).withIndex().associate { Pair(reducedScope[it.index], it.value) }
        val augmentedEvidenceIndex = assignmentToIndex(relevantEvidence + newScopeAssignment, fullFactor.scope)
        reducedProbTable[i] = fullFactor.values[augmentedEvidenceIndex]
    }

    return Factor(reducedScope, reducedProbTable.toList())

}

fun applyEvidence(evidence : Map<RandomVariable, Int>, fullFactor : Factor) : Factor{
    val relevantEvidence = evidence.filterKeys { fullFactor.scope.contains(it) }
    val relevantEvidenceScope = relevantEvidence.keys.toList()
    val evidenceAssignmentIndex = assignmentToIndex(relevantEvidence, relevantEvidenceScope)

    val indicatorProbs = (0 until numAssignments(relevantEvidenceScope)).map { if (it == evidenceAssignmentIndex) 1.0 else 0.0 }
    val indicatorFactor = Factor(relevantEvidenceScope, indicatorProbs)
    return product(indicatorFactor, fullFactor)
}


/*
data class CalibratedCliqueForest(val cliqueTrees : List<CalibratedCliqueTree>){
    val cliqueBeliefs = cliqueTrees.map { it.cliqueBeliefs }.reduce { accBeliefMap, beliefMap -> accBeliefMap + beliefMap }
    val sepsetBeliefs = cliqueTrees.map { it.sepsetBeliefs }.reduce { accSetsetMap, sepsetMap -> accSetsetMap + sepsetMap }
}

data class CalibratedCliqueTree(val cliqueBeliefs : Map<CliqueNode, Factor>,
                                val sepsetBeliefs : Map<Set<CliqueNode>, Factor>){
    override fun toString() = "Clique Beliefs: \n${cliqueBeliefs.values.joinToString("\n")}\n Sepset Beliefs: \n${sepsetBeliefs.values.joinToString("\n")}"
}

fun assignPotentials(cliqueForest : List<CliqueTree>, factors : List<Factor>) : Map<CliqueNode, List<Factor>>{
    val cliqueNodes = cliqueForest.flatMap{ it.nodes }
    val potentialAssignment = HashMap(cliqueNodes.associate { Pair(it, ArrayList<Factor>()) })

    for(factor in factors) {
        val factorScope = factor.scope.toSet()
        val assignedClique = cliqueNodes.find { it.scope.containsAll(factorScope) }
        potentialAssignment[assignedClique]!!.add(factor)
    }
    return potentialAssignment
}

fun inCliqueQuery(queryVars : List<RandomVariable>, calibratedForest: CalibratedCliqueForest) : Factor {
    val relevantBelief = calibratedForest.cliqueBeliefs.values.find { it.scope.containsAll(queryVars) } ?:
        throw IllegalArgumentException("No single clique contains $queryVars. Out of clique queries not yet supported")

    val irrelevantVars = relevantBelief.scope.toSet() - queryVars.toSet()
    return normalize(sumOut(relevantBelief, irrelevantVars.toList()))
}

fun outOfCliqueQuery(queryVars : List<RandomVariable>, calibratedForest: CalibratedCliqueForest) : Factor{
    val factors = relevantBeliefFactors(queryVars, calibratedForest)
    val irrelevantVars = factors.flatMap { it.scope } - queryVars
    return reorderFactor(normalize(variableElimination(factors, irrelevantVars)), queryVars)
}

fun relevantBeliefFactors(queryVars : List<RandomVariable>, calibratedForest : CalibratedCliqueForest) : List<Factor> {
    val singleFactor = calibratedForest.cliqueBeliefs.values.find { it.scope.containsAll(queryVars)}
    if(singleFactor != null){
        return listOf(singleFactor)
    }

    val relevantNodesByTree = calibratedForest.cliqueTrees.map {
        it.cliqueBeliefs.keys.filter { (it.scope intersect queryVars).isNotEmpty() }
    }.filter { it.isNotEmpty() }

    val doubleIndependentClique = findPairFromSeparateGroups(relevantNodesByTree, {c1, c2 -> (c1.scope + c2.scope).containsAll(queryVars)})
    if(doubleIndependentClique != null){
        return doubleIndependentClique.toList().map { calibratedForest.cliqueBeliefs[it]!! }
    }

    // How about two factors within the same tree? Find the one with the shortest path!
    val bestPath = relevantNodesByTree.flatten()
        .map { shortestPathContainingScope(queryVars, it) }
        .filterNotNull()
        .minBy { it.size } ?:
        throw IllegalArgumentException("Query not answerable by 1 or 2 factors. Additional factor queries not yet supported")

    return uncalibratePath(bestPath, calibratedForest.cliqueBeliefs, calibratedForest.sepsetBeliefs)
}

fun uncalibratePath(cliqueNodePath : List<CliqueNode>, cliqueBeliefs : Map<CliqueNode, Factor>, sepsetBeliefs: Map<Set<CliqueNode>, Factor>) =
    cliqueNodePath.indices.map { i ->
        if(i == cliqueNodePath.size - 1)
            cliqueBeliefs[cliqueNodePath[i]]!!
        else{
            val factorNumerator = cliqueBeliefs[cliqueNodePath[i]]!!
            val factorDenominator = sepsetBeliefs[setOf(cliqueNodePath[i], cliqueNodePath[i + 1])]!!
            divide(factorNumerator, factorDenominator)
        }

    }

fun shortestPathContainingScope(queryVars : List<RandomVariable>, startingNode : CliqueNode) : List<CliqueNode>? {
    fun shortestPathRec(currentNode : CliqueNode, accumulatedScope : Set<RandomVariable>, pathSoFar : List<CliqueNode>, bestPath : List<CliqueNode>?) : List<CliqueNode>?{
        val scope = accumulatedScope + currentNode.scope

        if(bestPath != null && pathSoFar.size >= bestPath.size)
            return bestPath
        if(scope.containsAll(queryVars))
            return pathSoFar

        return currentNode.neighbors
            .filter { !pathSoFar.contains(it) }
            .fold(bestPath, {currentBest, node -> shortestPathRec(node, scope, pathSoFar + node, currentBest)}) ?: bestPath
    }

    return shortestPathRec(startingNode, emptySet(), listOf(startingNode), null)
}

fun initialisePotentials(potentialAssignment: Map<CliqueNode, List<Factor>>) =
    potentialAssignment.mapValues{ (_, factors) ->
        if(factors.isEmpty())
            Factor(emptyList(), listOf(1.0))
        else
            factors.reduce(::product)
    }

fun sumProductCalibrate(cliqueTree: CliqueTree, potentialAssigment : Map<CliqueNode, List<Factor>>) : CalibratedCliqueTree {
    val combinedPotentials = initialisePotentials(potentialAssigment)
    //println("Combined Potentials: \n${combinedPotentials.values.joinToString("\n")}")
    val rootNode = cliqueTree.nodes.first()

    val upwardSendPath = pathToRoot(rootNode)
    val downwardSendPaths = pathsFromRoot(rootNode)

    val incomingMessages = cliqueTree.nodes.associate { Pair(it, ArrayList<Message>()) }
    val notYetSenders = ArrayList(cliqueTree.nodes)

    // Upward Pass
    //println("Upward Pass")
    while(incomingMessages[rootNode]!!.size < rootNode.neighbors.size){
        val sentThisRound = ArrayList<CliqueNode>()
        for(sender in notYetSenders){
            if(sender != rootNode && incomingMessages[sender]!!.size == sender.neighbors.size - 1){
                val receiver = upwardSendPath[sender]!!
                val messageFactor = sendMessage(sender, receiver, incomingMessages[sender]!!, combinedPotentials)
                sentThisRound.add(sender)
                incomingMessages[receiver]!!.add(messageFactor)
            }
        }
        notYetSenders.removeAll(sentThisRound)
    }

    // Downward Pass
    //println("Downward Pass")
    val downSendFrontier = mutableListOf(rootNode)
    while(downSendFrontier.isNotEmpty()){
        val sender = downSendFrontier.removeAt(downSendFrontier.size - 1)
        for(receiver in downwardSendPaths[sender]!!){
            downSendFrontier.add(receiver)
            val message = sendMessage(sender, receiver, incomingMessages[sender]!!, combinedPotentials)
            incomingMessages[receiver]!!.add(message)
        }
    }

    val beliefFactors = HashMap<CliqueNode, Factor>()
    for(cliqueNode in cliqueTree.nodes){
        if(incomingMessages[cliqueNode]!!.isNotEmpty()){
            val finalMessageFactors = incomingMessages[cliqueNode]!!.map { it.factor }
            val finalMessageProduct = finalMessageFactors.reduce(::product)
            beliefFactors[cliqueNode] = product(combinedPotentials[cliqueNode]!!, finalMessageProduct)
        }
        else{
            beliefFactors[cliqueNode] = combinedPotentials[cliqueNode]!!
        }
    }

    val sepsets = HashMap<Set<CliqueNode>, Factor>()
    for(cliqueNode in upwardSendPath.keys){
        val varIntersection = cliqueNode.scope.toSet() intersect  upwardSendPath[cliqueNode]!!.scope.toSet()
        val sumOutVars = cliqueNode.scope.toSet() - varIntersection
        val sepset = sumOut(beliefFactors[cliqueNode]!!, sumOutVars.toList())
        sepsets[setOf(cliqueNode, upwardSendPath[cliqueNode]!!)] = sepset
    }

    return CalibratedCliqueTree(beliefFactors, sepsets)
}

fun sendMessage(sender : CliqueNode, receiver : CliqueNode, incomingSenderMessages : List<Message>, combinedPotentials : Map<CliqueNode, Factor>) : Message{
    //println("${sender.scope} sends message to ${receiver.scope}")
    val factors = incomingSenderMessages.filter { it.sender != receiver }.map { it.factor } + combinedPotentials[sender]!!
    val fullProduct = factors.reduce(::product)
    val varsToRemove = sender.scope.toSet() - (sender.scope.toSet() intersect  receiver.scope.toSet())
    val summedOutFactor = sumOut(fullProduct, varsToRemove.toList())
    return Message(sender, receiver, summedOutFactor)
}

fun pathToRoot(rootNode : CliqueNode) : Map<CliqueNode, CliqueNode>{
    val sendPath = HashMap<CliqueNode, CliqueNode>()
    val visitedNodes = mutableSetOf(rootNode)
    val frontier = mutableListOf(rootNode)
    while (frontier.isNotEmpty()){
        val node = frontier.removeAt(frontier.size - 1)
        val unvisitedNeighbors = node.neighbors.filter {!visitedNodes.contains(it)}
        for(neighbor in unvisitedNeighbors){
            sendPath[neighbor] = node
            frontier.add(neighbor)
            visitedNodes.add(neighbor)
        }
    }

    return sendPath
}

fun pathsFromRoot(rootNode : CliqueNode) : Map<CliqueNode, MutableList<CliqueNode>>{
    val sendPath = HashMap<CliqueNode, ArrayList<CliqueNode>>()
    val visitedNodes = mutableSetOf(rootNode)
    val frontier = mutableListOf(rootNode)

    while(frontier.isNotEmpty()){
        val node = frontier.removeAt(frontier.size - 1)
        val unvisitedNeighbors = node.neighbors.filter {!visitedNodes.contains(it)}
        frontier.addAll(unvisitedNeighbors)
        sendPath.getOrPut(node, {ArrayList()}).addAll(unvisitedNeighbors)
        visitedNodes.add(node)
    }

    return sendPath
}
*/
