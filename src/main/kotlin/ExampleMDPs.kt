data class MDP(val vocab : Set<RandomVariable>, val rewardTree : DecisionTree<Reward>, val actions : Set<Action>, val dbns : Map<Action, DynamicBayesNet>){
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

    return MDP(vocab, rewardTree, actions, dbns)
}

fun unifStartIDTransJointDBN(vocab : Set<RandomVariable>, noise : Double = 0.0) : DynamicBayesNet{
    return vocab.associate { rv -> Pair(rv, identityTransition(rv, noise)) }
}

fun identityTransitionDBN(vocab : Set<RandomVariable>, noise : Double = 0.0) : DynamicBayesNet {
    return vocab.associate { rv -> Pair(rv, identityTransition(rv, noise)) }
}