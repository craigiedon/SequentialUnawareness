import org.junit.*

class SVITests{
    val X = RandomVariable("X", 2)
    val Y = RandomVariable("Y", 2)
    val W = RandomVariable("W", 2)
    val Z = RandomVariable("Z", 2)

    val xCPD = identityTransition(X)
    val wCPD = identityTransition(W)
    val yCPD = DTDecision(Y, listOf(
                    DTDecision(X, listOf(
                        DTLeaf(detFactor(Y, 0)),
                        DTLeaf(Factor(listOf(Y), listOf(0.1, 0.9))))
                    ),
                    DTLeaf(detFactor(Y, 1)))
                )
    val zCPD = DTDecision(Z, listOf(
        DTDecision(Y, listOf(
            DTLeaf(detFactor(Z, 0)),
            DTLeaf(Factor(listOf(Z), listOf(0.1, 0.9))))
        ),
        DTLeaf(detFactor(Z, 1)))
    )

    val rewardTree = DTDecision(Z, listOf(DTLeaf(0.0), DTLeaf(10.0)))

    val dbn = DynamicBayesNet(mapOf(
        X to xCPD,
        Y to yCPD,
        Z to zCPD,
        W to wCPD
    ))

    val value1Step = DTDecision(Z, listOf(
        DTDecision(Y, listOf(
            DTLeaf(0.0),
            DTLeaf(8.1)
        )),
        DTLeaf(19.0)
    ))

    @Test
    fun identicalReward_emptyPTree(){
        val vTree = DTLeaf(1.0)
        val pTree = pRegress(vTree, dbn)


        Assert.assertTrue(checkEquality(DTLeaf(emptyMap<RandomVariable, Factor>()), pTree, ::doubleEquality))
    }

    @Test
    fun simplifiedTree_returnSameTree(){
        Assert.assertTrue(checkEquality(zCPD, simplify(zCPD, ::doubleEquality), ::doubleEquality))
    }

    @Test
    fun TreeApendedToIdenticalTree_originalTree(){
        val mapifiedZ = fMap(zCPD, {mapOf(it.scope.first() to it)})
        Assert.assertTrue(checkEquality(mapifiedZ, append(mapifiedZ, mapifiedZ, {m1, m2 -> m1 + m2}, ::doubleEquality), ::doubleEquality))
    }

    @Test
    fun pRegress_valueAtZeroSteps_dbnCPDTree(){
        val pTree = pRegress(rewardTree, dbn)
        Assert.assertTrue(checkEquality(fMap(zCPD, {mapOf(it.scope.first() to it)}), pTree, ::doubleEquality))
    }

    @Test
    fun futureValue_valueAtZeroSteps_FVTree(){
        val pTree = pRegress(rewardTree, dbn)
        val fvTree = undiscountedFutureValue(rewardTree, pTree)
        val expectedFVTree = DTDecision(Z, listOf(
            DTDecision(Y, listOf(
                DTLeaf(0.0),
                DTLeaf(9.0)
            )),
            DTLeaf(10.0)
        ))
        Assert.assertTrue(checkEquality(expectedFVTree, fvTree, ::doubleEquality))
    }

    @Test
    fun regress_initialReward_discountedValueTree(){
        val expected= DTDecision(Z, listOf(
            DTDecision(Y, listOf(
                DTLeaf(0.0),
                DTLeaf(8.1)
            )),
            DTLeaf(19.0)
        ))
        val result = regress(rewardTree, rewardTree, dbn)
        Assert.assertTrue(checkEquality(expected, result, ::doubleEquality))
    }

    @Test
    fun pRegress_valueAfter1step_pTree(){
        val expected = DTDecision(Z, listOf(
            DTDecision(Y, listOf(
                DTDecision(X, listOf(
                    DTLeaf(mapOf(Z to detFactor(Z, 0), Y to detFactor(Y, 0))),
                    DTLeaf(mapOf(Z to detFactor(Z, 0), Y to Factor(listOf(Y), listOf(0.1, 0.9)))))
                ),
                DTLeaf(mapOf(Z to Factor(listOf(Z), listOf(0.1, 0.9)), Y to detFactor(Y, 1)))
            )),
            DTLeaf(mapOf(Z to detFactor(Z, 1)))
        ))
        val result = pRegress(value1Step, dbn)

        Assert.assertTrue(checkEquality(expected, result, ::doubleEquality))
    }

    @Test
    fun futureValue_valueAfter1Step_FVTree(){
        val expected = DTDecision(Z, listOf(
            DTDecision(Y, listOf(
                DTDecision(X, listOf(
                    DTLeaf(0.0),
                    DTLeaf(7.29)
                )),
                DTLeaf(17.91)
            )),
            DTLeaf(19.0)
        ))
        val pTree = pRegress(value1Step, dbn)
        val result = undiscountedFutureValue(value1Step, pTree)
        Assert.assertTrue(checkEquality(expected, result, ::doubleEquality))
    }

    @Test
    fun regress_valueAfter1Step_discountedValueTree(){
        val expected = DTDecision(Z, listOf(
            DTDecision(Y, listOf(
                DTDecision(X, listOf(
                    DTLeaf(0.0),
                    DTLeaf(6.56)
                )),
                DTLeaf(16.119)
            )),
            DTLeaf(27.1)
        ))
        val result = regress(value1Step, rewardTree, dbn)
        Assert.assertTrue(checkEquality(expected, result, ::doubleEquality))
    }

}
