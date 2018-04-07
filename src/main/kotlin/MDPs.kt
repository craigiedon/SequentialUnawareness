import com.fasterxml.jackson.databind.JsonNode

data class MDP(val vocab : Set<RandomVariable>, val rewardTree : DecisionTree<Reward>, val actions : Set<Action>, val dbns : Map<Action, DynamicBayesNet>, val discount : Double){
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
    val yCPD = DTDecision(Y, listOf(
        DTDecision(X, listOf(
            DTLeaf(detFactor(Y, 0)),
            DTLeaf(Factor(listOf(Y), listOf(0.1, 0.9)))
        )),
        DTLeaf(detFactor(Y, 1))
    ))
    val zCPD = DTDecision(Z, listOf(
        DTDecision(Y, listOf(
            DTLeaf(detFactor(Z, 0)),
            DTLeaf(Factor(listOf(Z), listOf(0.1, 0.9)))
        )),
        DTLeaf(detFactor(Z, 1))
    ))

    val rewardTree = DTDecision(Z, listOf(
        DTLeaf(0.0),
        DTLeaf(10.0)
    ))

    val cpdTrees = mapOf(X to xCPD, Y to yCPD, W to wCPD, Z to zCPD)
    val actions = setOf("A1")
    val dbns = mapOf("A1" to cpdTrees)

    return MDP(vocab, rewardTree, actions, dbns, 0.99)
}

/*
fun unifStartIDTransJointDBN(vocab : Set<RandomVariable>, noise : Double = 0.0) : DynamicBayesNet{
    return vocab.associate { rv -> Pair(rv, identityTransition(rv, noise)) }
}
*/

fun identityTransitionDBN(vocab : Set<RandomVariable>, noise : Double = 0.0) : DynamicBayesNet {
    return vocab.associate { rv -> Pair(rv, identityTransition(rv, noise)) }
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
    return MDP(vocab.values.toSet(), rewardTree, actionNames, dbns, discount)
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
    val branches = decisionVar.domain.map { parseProbTree(decisionNode.get(it), rv, vocab) }
    return DTDecision(decisionVar, branches)
}

fun parseRewardTree(jsonRewardNode : JsonNode, vocab : Map<String, RandomVariable>) : DecisionTree<Reward> {
    if(jsonRewardNode.isArray && jsonRewardNode.size() == 1){
        return DTLeaf(jsonRewardNode[0].asDouble())
    }

    val (varName, decisionNode) = jsonRewardNode.fields().next()
    val rv = vocab[varName]!!
    val branches = rv.domain.map { parseRewardTree(decisionNode.get(it), vocab) }
    return DTDecision(rv, branches)
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