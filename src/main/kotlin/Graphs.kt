import java.util.*

val A = RandomVariable("A", 2)
val B = RandomVariable("B", 2)
val C = RandomVariable("C", 2)
val D = RandomVariable("D", 2)
val E = RandomVariable("E", 2)
val F = RandomVariable("F", 2)
val G = RandomVariable("G", 2)
val I = RandomVariable("I", 2)
val S = RandomVariable("S", 2)
val L = RandomVariable("L", 2)
val J = RandomVariable("J", 2)
val H = RandomVariable("H", 2)

data class RandomVariable(val name : String, val domainSize : Int, val varType : VarType = VarType.BELIEF){
    override fun toString() = name
    val domain get() = (0..domainSize - 1).toList()
}

enum class VarType { BELIEF, ACTION, OUTCOME }

interface DirNode<out V>{
    fun item() : V
    fun parents() : List<V>
}

data class PlainNode<out V>(val item : V, val parents : List<V>) : DirNode<V>{
    override fun item() = item
    override fun parents() = parents
}

data class BNode(val rv : RandomVariable, val parents : List<RandomVariable>, val cpt : List<Double>, val hidden : Boolean = false) : DirNode<RandomVariable>{
    override fun item() = rv
    override fun parents() = parents
    override fun toString(): String = "$rv - parents: $parents cpt:$cpt hidden?:$hidden"
}

typealias BNStruct = Map<RandomVariable, Set<RandomVariable>>

class BayesNet(val nodes : Map<RandomVariable, BNode>){
    override fun toString() = nodes.values.joinToString("\n")
    val vocab : Set<RandomVariable> get() = nodes.keys
    val observableVars : List<RandomVariable> get() = nodes.keys.filter { !nodes[it]!!.hidden }
    val hiddenVars : List<RandomVariable> get() = nodes.keys.filter { nodes[it]!!.hidden }
}

data class BNodeData(val name: String, val parents : List<String>, val cpt: List<Double>, val domainSize: Int)

class Node(val rv : RandomVariable, val neighbors: HashSet<Node>)
data class Graph(val nodes : Map<RandomVariable, Node>)

fun removeVar(bnStruct : BNStruct, rvToRemove: RandomVariable) : BNStruct{
    if(!bnStruct.containsKey(rvToRemove))
        throw IllegalArgumentException("RV not contained in structure")
    val rvParents = bnStruct[rvToRemove]!!
    val rvChildren = bnStruct.keys.filter { bnStruct[it]!!.contains(rvToRemove) }
    val newStruct = HashMap(bnStruct)
    newStruct.remove(rvToRemove)
    for((i, child) in rvChildren.withIndex()){
        newStruct[child] = newStruct[child]!! - rvToRemove + rvChildren.subList(i + 1, rvChildren.size) + rvParents
    }
    return newStruct
}

fun removeVars(bnStruct: BNStruct, rvsToRemove : List<RandomVariable>) =
    rvsToRemove.fold(bnStruct, ::removeVar)

fun markovBlanket(targetRV: RandomVariable, parentStructure : Map<RandomVariable, Set<RandomVariable>>) : Map<RandomVariable, Set<RandomVariable>> {
    val childrensParents = parentStructure.filterKeys { possibleChild -> parentStructure[possibleChild]!!.contains(targetRV) }
    val parentPair = Pair(targetRV, parentStructure[targetRV]!!)
    return childrensParents + parentPair
}

fun markovBlankets(targetRVs : List<RandomVariable>, parentStructure : Map<RandomVariable, Set<RandomVariable>>) : Map<RandomVariable, Set<RandomVariable>> {
    val markovBlankets = HashMap<RandomVariable, Set<RandomVariable>>()
    for(targetRV in targetRVs){
        val markovBlanket = markovBlanket(targetRV, parentStructure)
        for((rv, knownParents) in markovBlanket) {
            markovBlankets[rv] = (markovBlankets[rv] ?: emptySet()) + knownParents
        }
    }
    return markovBlankets
}

fun parentMapToChildMap(parentMap : Map<RandomVariable, Set<RandomVariable>>) =
    parentMap.mapValues{
        (parentRV, _) -> parentMap.keys.filter{ parentMap[it]!!.contains(parentRV) }.toSet()
    }

fun parentChildSubstructure(bnStruct : BNStruct, targetRVs : List<RandomVariable>) : BNStruct{
    // Create a substructure which contains only the target variables and their parents/children
    val childMap = parentMapToChildMap(bnStruct)
    val subStruct = bnStruct.filterKeys{targetRVs.contains(it)}.toMutableMap()
    for(targetRV in targetRVs){
        for(child in childMap[targetRV]!!){
            subStruct[child] = (subStruct[child] ?: emptySet()) + targetRV
        }
    }
    return subStruct
}

fun moralize(bn : BayesNet) : Graph{
    val nodeMap = bn.nodes.mapValues {Node(it.key, HashSet())}

    for (node in bn.nodes.values){
        val parentNeighbors = node.parents.map { nodeMap[it]!! }
        nodeMap[node.rv]!!.neighbors.addAll(parentNeighbors)

        for(parent in node.parents){
            nodeMap[parent]!!.neighbors.addAll(parentNeighbors.filter { it.rv != parent })
            nodeMap[parent]!!.neighbors.add(nodeMap[node.rv]!!)
        }
    }

    return Graph(nodeMap)
}

fun <X> invert(parentSetStructure: Map<X, Set<X>>) : Map<X, Set<X>>{
    val childSetStructure = parentSetStructure.keys.associate { Pair(it, HashSet<X>()) }
    for((rv, parents) in parentSetStructure){
        for(parent in parents){
            childSetStructure[parent]!!.add(rv)
        }
    }
    return childSetStructure
}

fun reachable(parentSetStructure: Map<RandomVariable, Set<RandomVariable>>, sourceVar : RandomVariable, evidenceVars : Set<RandomVariable>) : Set<RandomVariable>{
    val ancestors = originalAndAncestors(parentSetStructure, evidenceVars)
    val childStructure = invert(parentSetStructure)

    // Traverse active trails starting from source
    val toVisit = LinkedList(listOf(Pair(sourceVar, "up")))
    val visited = HashSet<Pair<RandomVariable, String>>()
    val reachable = HashSet<RandomVariable>()
    while(toVisit.isNotEmpty()){
        val (rv, dir) = toVisit.pop()
        if(!visited.contains(Pair(rv,dir))){
            visited.add(Pair(rv, dir))


            if(!evidenceVars.contains(rv)){
                reachable.add(rv)
            }

            if(dir == "up" && !evidenceVars.contains(rv)){
                toVisit.addAll(parentSetStructure[rv]!!.map { Pair(rv, "up") })
                toVisit.addAll(childStructure[rv]!!.map{Pair(rv, "down")})
            }
            else if(dir == "down"){
                if(!evidenceVars.contains(rv)){
                    toVisit.addAll(childStructure[rv]!!.map{Pair(rv, "down")})
                }
                if(ancestors.contains(rv)){
                    toVisit.addAll(parentSetStructure[rv]!!.map{Pair(rv, "up")})
                }
            }
        }
    }
    return reachable
}

fun originalAndAncestors(bnStructure: Map<RandomVariable, Set<RandomVariable>>, rvs : Set<RandomVariable>) : Set<RandomVariable>{
    val toVisit = LinkedList(rvs)
    val ancestors = HashSet<RandomVariable>()
    // Insert all ancestors of evidence vars into list
    while(toVisit.isNotEmpty()){
        val currentVar = toVisit.pop()
        if(!ancestors.contains(currentVar)){
            toVisit.addAll(bnStructure[currentVar]!!)
        }
        ancestors.add(currentVar)
    }
    return ancestors
}

fun triangulate(moralizedGraph: Graph) : Pair<Map<RandomVariable, Int>, Graph>{
    val eliminationOrdering = HashMap<RandomVariable, Int>()
    val triangulatedNodeMap : Map<RandomVariable, Node> = moralizedGraph.nodes.mapValues { Node(it.key, HashSet(it.value.neighbors)) }
    for (i in 0..triangulatedNodeMap.size - 1){
        val unusedRVs : List<RandomVariable> = triangulatedNodeMap.keys.filter { !eliminationOrdering.containsKey(it) }
        val laterNeighbors = unusedRVs.map { Pair(it, triangulatedNodeMap[it]!!.neighbors.filter { n -> !eliminationOrdering.containsKey(n.rv) }) }
        val (minRV, minLaterNeighbors) = laterNeighbors.minBy { it.second.size }!!

        eliminationOrdering[minRV] = i
        for (neighborA in minLaterNeighbors){
            for(neighborB in minLaterNeighbors){
                if(neighborA != neighborB){
                    triangulatedNodeMap[neighborA.rv]!!.neighbors.add(neighborB)
                    triangulatedNodeMap[neighborB.rv]!!.neighbors.add(neighborA)
                }
            }
        }
    }
    return Pair(eliminationOrdering, Graph(triangulatedNodeMap))
}

fun <T, V : DirNode<T>> topologicalSort(unsortedNodes: List<V>): List<V> {
    val sortedNodes = ArrayList<V>()
    val sortVocab = unsortedNodes.map { it.item() }

    val incomingEdgesMap = unsortedNodes
        .associateBy(
            {it},
            {ArrayList(it.parents().filter { p -> p in sortVocab })}) as MutableMap
    var noIncomingEdgesQueue = unsortedNodes.filter{ it.parents().isEmpty() }

    while(noIncomingEdgesQueue.isNotEmpty()){
        sortedNodes.addAll(noIncomingEdgesQueue)
        for(node in noIncomingEdgesQueue){
            incomingEdgesMap.remove(node)
            for(otherNode in incomingEdgesMap.keys) {
                incomingEdgesMap[otherNode]!!.remove(node.item())
            }
        }
        noIncomingEdgesQueue = incomingEdgesMap.filter{it.value.size <= 0}.map{it.key}
    }

    if (incomingEdgesMap.isNotEmpty()) {throw IllegalStateException("Disconnected Nodes")}

    return sortedNodes
}

fun extractImmoralityEdges(bnStructure : Map<RandomVariable, Set<RandomVariable>>) : Map<RandomVariable, Set<RandomVariable>>{
    val immoralityParents = bnStructure.mapValues { HashSet<RandomVariable>() }
    for((rv, parents) in bnStructure){
        val orderedParents = parents.toList()
        for(pIndA in parents.indices){
            for(pIndB in (pIndA+1)..(parents.size - 1)){
                val parentA = orderedParents[pIndA]
                val parentB = orderedParents[pIndB]
                if(!bnStructure[parentA]!!.contains(parentB) && !bnStructure[parentB]!!.contains(parentA)){
                    immoralityParents[rv]!!.add(parentA)
                    immoralityParents[rv]!!.add(parentB)
                }
            }
        }
    }
    return immoralityParents
}

fun getSkeleton(bnStructure : Map<RandomVariable, Set<RandomVariable>>) : Map<RandomVariable, Set<RandomVariable>>{
    val undirectedSkeleton = bnStructure.mapValues { HashSet<RandomVariable>() }
    for((rv, parents) in bnStructure){
        undirectedSkeleton[rv]!!.addAll(parents)
        for(parent in parents){
            undirectedSkeleton[parent]!!.add(rv)
        }
    }
    return undirectedSkeleton
}

data class R1Match(val rv : RandomVariable, val parent : RandomVariable, val neighbour : RandomVariable)
fun matchR1PDag(rv : RandomVariable, parentStructure : Map<RandomVariable, Set<RandomVariable>>, undirectedEdges: Map<RandomVariable, Set<RandomVariable>>) : R1Match?{
    for (parent in parentStructure[rv]!!){
        for(neighbour in undirectedEdges[rv]!!){
            if(!undirectedEdges[parent]!!.contains(neighbour) && !parentStructure[parent]!!.contains(neighbour) && !parentStructure[neighbour]!!.contains(parent)){
                return R1Match(rv, parent, neighbour)
            }
        }
    }
    return null
}

data class R2Match(val rv : RandomVariable, val parent : RandomVariable, val parentNeighbour : RandomVariable)
fun matchR2PDAG(rv : RandomVariable, parentStructure : Map<RandomVariable, Set<RandomVariable>>, undirectedEdges: Map<RandomVariable, Set<RandomVariable>>) : R2Match?{
    for(parent in parentStructure[rv]!!){
        for(parentNeighbour in undirectedEdges[parent]!!){
            if(parentStructure[parentNeighbour]!!.contains(rv)){
                return R2Match(rv, parent, parentNeighbour)
            }
        }
    }

    return null
}

data class R3Match(val rv : RandomVariable, val p1 : RandomVariable, val p2 : RandomVariable, val commonNeighbour : RandomVariable)
fun matchR3PDAG(rv : RandomVariable, parentStructure: Map<RandomVariable, Set<RandomVariable>>, undirectedEdges: Map<RandomVariable, Set<RandomVariable>>) : R3Match?{
    // Apply rule 3
    // If rv has two unique parents
    val parents = parentStructure[rv]!!.toList()
    for(parentIndA in parents.indices){
        for(parentIndB in (parentIndA + 1)..(parents.size - 1)){
            val parentA = parents[parentIndA]
            val parentB = parents[parentIndB]
            val sharedNeighbours = undirectedEdges[parentA]!! intersect  undirectedEdges[parentB]!!
            val commonNeighbour = sharedNeighbours.find { undirectedEdges[it]!!.contains(rv) }
            if(commonNeighbour != null){
                return R3Match(rv, parentA, parentB, commonNeighbour)
            }
        }
    }
    return null
}

typealias PDAG = Map<RandomVariable, PDNode>
data class PDNode(val rv : RandomVariable, val directedParents: Set<RandomVariable>, val undirectedEdges: Set<RandomVariable>)

fun buildPDAG(bn : BayesNet)  = buildPDAG(rawStructure(bn))
fun buildPDAG(parentStructure : Map<RandomVariable, Set<RandomVariable>>) = buildPDAG(extractImmoralityEdges(parentStructure), getSkeleton(parentStructure))
fun buildPDAG(originalImmoralities: Map<RandomVariable, Set<RandomVariable>>, skeleton : Map<RandomVariable, Set<RandomVariable>>) : PDAG{
    val directedParents = HashMap(originalImmoralities)

    // Free undirected edges are the skeleton minus the directed edges we know about so far
    val undirectedEdges = HashMap(skeleton)
    for((rv, neighbours) in skeleton){
        undirectedEdges[rv] = neighbours.filter { !(directedParents[rv]!!.contains(it) || directedParents[it]!!.contains(rv)) }.toMutableSet()
    }

    do{
        var changed = false
        for(rv in directedParents.keys){
            val r1Result = matchR1PDag(rv, directedParents, undirectedEdges)
            if(r1Result != null){
                changed = true
                undirectedEdges[r1Result.neighbour] = undirectedEdges[r1Result.neighbour]!! - rv
                undirectedEdges[rv] = undirectedEdges[rv]!! - r1Result.neighbour
                directedParents[r1Result.neighbour] = directedParents[r1Result.neighbour]!! + rv
            }
            val r2Result = matchR2PDAG(rv, directedParents, undirectedEdges)
            if(r2Result != null){
                changed = true
                undirectedEdges[r2Result.parent] = undirectedEdges[r2Result.parent]!! - r2Result.parentNeighbour
                undirectedEdges[r2Result.parentNeighbour] = undirectedEdges[r2Result.parent]!! - r2Result.parent
                directedParents[r2Result.parentNeighbour] = directedParents[r2Result.parentNeighbour]!! + r2Result.parent
            }
            val r3Result = matchR3PDAG(rv, directedParents, undirectedEdges)
            if(r3Result != null){
                changed = true
                undirectedEdges[r3Result.commonNeighbour] = undirectedEdges[r3Result.commonNeighbour]!! - rv
                undirectedEdges[rv] = undirectedEdges[rv]!! - r3Result.commonNeighbour
                directedParents[rv] = directedParents[rv]!! + r3Result.commonNeighbour
            }
        }
    }
    while(changed)

    return skeleton.mapValues { PDNode(it.key, directedParents[it.key]!!, undirectedEdges[it.key]!!) }
}

@JvmName("SHDPDAG")
fun structuralHammingDistance(correctPDAG: PDAG, differingPDAG: PDAG) : Int{
    if(correctPDAG.keys != differingPDAG.keys){
        val missingFromCorrect = differingPDAG.keys - correctPDAG.keys
        val missingFromDiffering = correctPDAG.keys - differingPDAG.keys

        val augmentedCorrectPDAG = correctPDAG + missingFromCorrect.associate { Pair(it, PDNode(it, emptySet(), emptySet())) }
        val augmentedDifferingPDAG = differingPDAG + missingFromDiffering.associate{ Pair(it, PDNode(it, emptySet(), emptySet()))}
        return structuralHammingDistance(augmentedCorrectPDAG, augmentedDifferingPDAG)
    }

    val vocab = correctPDAG.keys.toList()

    var directedMistakes = 0
    var undirectedMistakes = 0
    for(rv in vocab){
        val correctParents = correctPDAG[rv]!!.directedParents
        val differingParents = differingPDAG[rv]!!.directedParents

        val missingParents = correctParents.filter { !differingParents.contains(it) }
        val extraParents = differingParents.filter { !correctParents.contains(it) }
        val wrongDirection = missingParents.filter {  differingPDAG[it]!!.directedParents.contains(rv)}

        val correctEdges = correctPDAG[rv]!!.undirectedEdges
        val differingEdges = differingPDAG[rv]!!.undirectedEdges

        // If there was some directed edge that matches this undirect edge, then it will have already been counted
        val missingEdges = correctEdges.filter { !differingEdges.contains(it) && !differingPDAG[rv]!!.directedParents.contains(it) && !differingPDAG[it]!!.directedParents.contains(rv) }
        val extraEdges = differingEdges.filter { !correctEdges.contains(it) && !correctPDAG[rv]!!.directedParents.contains(it) && !correctPDAG[it]!!.directedParents.contains(rv) }

        directedMistakes += missingParents.size + extraParents.size - wrongDirection.size
        undirectedMistakes += missingEdges.size + extraEdges.size
    }

    return directedMistakes + (undirectedMistakes / 2)
}

fun rawStructure(bn : BayesNet) : Map<RandomVariable, Set<RandomVariable>> = bn.nodes.mapValues { it.value.parents.toSet() }

fun structuralHammingDistance(struct1: BNStruct, struct2: BNStruct) = structuralHammingDistance(buildPDAG(struct1), buildPDAG(struct2))
fun structuralHammingDistance(bn1: BayesNet, bn2: BayesNet) = structuralHammingDistance(buildPDAG(bn1), buildPDAG(bn2))

fun pruneIneffectualVars(bnStruct : BNStruct) : List<RandomVariable>{
    val childMap = parentMapToChildMap(bnStruct)
    return bnStruct.keys.filter { rv ->
        !childMap[rv]!!.isEmpty() && !(bnStruct[rv]!!.isEmpty() && childMap[rv]!!.size <= 1)
    }
}

fun hideNodes(bn : BayesNet, varsToHide : List<RandomVariable>) =
    BayesNet(bn.nodes.mapValues { (rv, node) ->
        if(varsToHide.contains(rv)) node.copy(hidden = true) else node
    })

@JvmName("HideNodesByName")
fun hideNodes(bn : BayesNet, varsToHide : List<String>) =
    hideNodes(bn, bn.vocab.filter { varsToHide.contains(it.name) })
