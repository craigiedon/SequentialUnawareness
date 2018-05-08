import com.fasterxml.jackson.databind.JsonNode

data class MDP(val vocab : Set<RandomVariable>,
               val rewardTree : DecisionTree<Reward>,
               val actions : Set<Action>,
               val dbns : Map<Action, DynamicBayesNet>,
               val discount : Double,
               val startStateDescriptions : List<RVAssignment> = emptyList(),
               val terminalDescriptions : List<RVAssignment> = emptyList()){
    val dbnStructs : Map<Action, DBNStruct> get() = dbns.mapValues { dbnStruct(it.value) }
    val rewardScope : Set<RandomVariable> get() = vocabInDT(rewardTree)
}

fun dbnStruct(dbn : DynamicBayesNet) =
    dbn.mapValues { (_, dt) -> vocabInDT(dt) }


fun simpleBoutillier() : MDP {
    val X = RandomVariable("X", 2)
    val Y = RandomVariable("Y", 2)
    val W = RandomVariable("W", 2)
    val Z = RandomVariable("Z", 2)
    val vocab = setOf(X, Y, W, Z)

    val xCPD = identityTransition(X)
    val wCPD = identityTransition(W)
    val yCPD = DTDecision(Pair(Y,0),
        DTDecision(Pair(X,0),
            DTLeaf(detFactor(Y, 0)),
            DTLeaf(Factor(listOf(Y), listOf(0.1, 0.9)))
        ),
        DTLeaf(detFactor(Y, 1))
    )
    val zCPD = DTDecision(Pair(Z,0),
        DTDecision(Pair(Y,0),
            DTLeaf(detFactor(Z, 0)),
            DTLeaf(Factor(listOf(Z), listOf(0.1, 0.9)))
        ),
        DTLeaf(detFactor(Z, 1))
    )

    val rewardTree = DTDecision(Pair(Z,0),
        DTLeaf(0.0),
        DTLeaf(10.0)
    )

    val cpdTrees = mapOf(X to xCPD, Y to yCPD, W to wCPD, Z to zCPD)
    val actions = setOf("A1")
    val dbns = mapOf("A1" to cpdTrees)

    return MDP(vocab, rewardTree, actions, dbns, 0.99)
}

fun loadMDP(jsonFilePath : String) : MDP{
    val jsonNode = loadJsonTree(jsonFilePath)
    // 1. Extract and create the random variables
    val variableJSON = jsonNode.get("variables")
    val vocab = HashMap<String, RandomVariable>()
    for ((varName, varDomain) in variableJSON.fields()){
        val domain  = varDomain.toList().map {it.asText()}
        vocab[varName] = RandomVariable(varName, domain)
    }

    // 2. For each action, extract a decision tree for each variable's transition function
    val actionsJSON = jsonNode.get("actions")
    val actionNames = actionsJSON.fieldNames().asSequence().toSet()
    val dbns = actionsJSON.fields().asSequence()
        .associate { (aName, dts) -> Pair(aName, dts.fields().asSequence()
            .associate { (varName, dtNode)  ->
                val rv = vocab[varName]!!
                Pair(rv, parseProbTree(dtNode, rv, vocab))
            })
        }


    val rewardTree = parseRewardTree(jsonNode.get("reward"), vocab)

    val discount : Double = jsonNode.get("discount").asDouble()

    val terminalDescriptions = extractStateDescriptions(jsonNode, vocab, "terminalDescriptions")
    val startStateDescriptions = extractStateDescriptions(jsonNode, vocab, "startStates")

    return MDP(vocab.values.toSet(), rewardTree, actionNames, dbns, discount, startStateDescriptions, terminalDescriptions)
}

fun extractStateDescriptions(jsonNode : JsonNode, vocab : Map<String, RandomVariable>, fieldName : String) : List<RVAssignment>{
    if(!jsonNode.has(fieldName)){
        return emptyList()
    }

    return jsonNode.get(fieldName)
        .asSequence()
        .map { stateDescription ->
            stateDescription.fields()
                .asSequence()
                .associate { (rvString, nodeVal) ->
                    val rv = vocab[rvString]!!
                    val domVal = nodeVal.asText()
                    Pair(rv, rv.domain.indexOf(domVal))
                }
        }.toList()
}

fun parseProbTree(jsonProbNode : JsonNode, rv : RandomVariable, vocab : Map<String, RandomVariable>) : DecisionTree<Factor>{
    if(jsonProbNode.isArray){
        val probs = jsonProbNode
            .asSequence()
            .map { it.asDouble() }
            .toList()

        return DTLeaf(Factor(listOf(rv), probs))
    }
    val (varName, decisionNode) = jsonProbNode.fields().next()
    val decisionVar = vocab[varName]!!
    val branches = decisionVar.domain.withIndex().associate { (i, domVal) -> Pair(i, parseProbTree(decisionNode.get(domVal), rv, vocab)) }
    return dtFromBranchMap(decisionVar, branches)
}

fun parseRewardTree(jsonRewardNode : JsonNode, vocab : Map<String, RandomVariable>) : DecisionTree<Reward> {
    if(jsonRewardNode.isArray && jsonRewardNode.size() == 1){
        return DTLeaf(jsonRewardNode[0].asDouble())
    }

    val (varName, decisionNode) = jsonRewardNode.fields().next()
    val rv = vocab[varName]!!
    val branches = rv.domain.withIndex().associate { (i, domVal) -> Pair(i, parseRewardTree(decisionNode.get(domVal), vocab)) }
    return dtFromBranchMap(rv, branches)
}

fun <T> dtFromBranchMap(rv : RandomVariable, branchMap : Map<Int, DecisionTree<T>>): DecisionTree<T>{
    fun dtFromBranchMapRec(branchesLeft : List<Int>) : DecisionTree<T>{
        if(branchesLeft.size == 1){
            return branchMap[branchesLeft[0]]!!
        }
        return DTDecision(Pair(rv, branchesLeft[0]), branchMap[branchesLeft[0]]!!, dtFromBranchMapRec(branchesLeft.drop(1)))
    }
    return dtFromBranchMapRec(branchMap.keys.toList())
}

fun generateInitialState(vocab : List<RandomVariable>, startStateDescriptions: List<RVAssignment>, terminalDescriptions : List<RVAssignment>) : RVAssignment {
    if(terminalDescriptions.any{ !vocab.containsAll(it.keys)} || startStateDescriptions.any { !vocab.containsAll(it.keys) }){
        throw IllegalArgumentException("State Descriptions contain vocabulary not present in given vocab")
    }

    while(true){
        val potentialSample = generateSample(vocab)
        val nonTerminal = terminalDescriptions.none { td -> partialMatch(td, potentialSample) }
        val matchesStartRequirement = startStateDescriptions.isEmpty() || startStateDescriptions.any { sd -> partialMatch(sd, potentialSample) }

        if(nonTerminal && matchesStartRequirement){
            return potentialSample
        }
    }
}

fun main(args : Array<String>) {
    val mdp = loadMDP("mdps/coffee.json")
    println(mdp.actions)
    println(mdp.vocab)
    println(mdp.discount)
    println(mdp.rewardScope)
    println(mdp.rewardTree)
    println(mdp.dbnStructs)
    println(mdp.dbns)
    println(mdp)
}