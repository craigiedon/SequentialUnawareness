fun boutillierSimple() : Triple<Map<Action, DynamicBayesNet>, DecisionTree<Reward>, Set<RandomVariable>>{
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

    val cpdTrees = mapOf(X to xCPD, Y to yCPD, W to wCPD, Z to zCPD)

    val rewardTree = DTDecision(Z, listOf(
        DTLeaf(0.0),
        DTLeaf(10.0)
    ))
    val trueDBNs = mapOf("A1" to cpdTrees)
    return Triple(trueDBNs, rewardTree, vocab)
}
