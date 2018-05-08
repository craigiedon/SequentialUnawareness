import org.junit.*

class SVITests{
    val X = RandomVariable("X", 2)
    val Y = RandomVariable("Y", 2)
    val W = RandomVariable("W", 2)
    val Z = RandomVariable("Z", 2)
    val T = RandomVariable("T", listOf("t1", "t2", "t3"))

    val xCPD = identityTransition(X)
    val wCPD = identityTransition(W)
    val yCPD = DTDecision(Pair(Y, 0),
                    DTDecision(Pair(X, 0),
                        DTLeaf(detFactor(Y, 0)),
                        DTLeaf(Factor(listOf(Y), listOf(0.1, 0.9)))
                    ),
                    DTLeaf(detFactor(Y, 1))
                )
    val zCPD = DTDecision(Pair(Z, 0),
        DTDecision(Pair(Y, 0),
            DTLeaf(detFactor(Z, 0)),
            DTLeaf(Factor(listOf(Z), listOf(0.1, 0.9)))
        ),
        DTLeaf(detFactor(Z, 1))
    )

    val rewardTree = DTDecision(Pair(Z, 0), DTLeaf(0.0), DTLeaf(10.0))

    val dbn = mapOf(
        X to xCPD,
        Y to yCPD,
        Z to zCPD,
        W to wCPD
    )

    val value1Step = DTDecision(Pair(Z, 0),
        DTDecision(Pair(Y, 0),
            DTLeaf(0.0),
            DTLeaf(8.1)
        ),
        DTLeaf(19.0)
    )

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
    fun merge_naiveFactorsSameStructure_returnSingle(){
        val dt1 = DTDecision(Pair(Y, 0),
            DTDecision(Pair(X, 0),
                DTLeaf(mapOf(
                    X to Factor(listOf(X), listOf(0.1, 0.9)),
                    Y to Factor(listOf(Y), listOf(0.6, 0.4)))
                ),
                DTLeaf(mapOf(
                    Y to Factor(listOf(Y), listOf(0.4,0.6)),
                    X to Factor(listOf(X), listOf(0.9, 0.1))
                ))
            ),
            DTDecision(Pair(X, 0),
                DTLeaf(mapOf(
                    X to Factor(listOf(X), listOf(0.2, 0.8)),
                    Y to Factor(listOf(Y), listOf(0.7, 0.3))
                )),
                DTLeaf(mapOf(
                    X to Factor(listOf(X), listOf(0.8, 0.2)),
                    Y to Factor(listOf(Y), listOf(0.3, 0.7))
                ))
            )
        )

        val dt2 = dt1
        val result = merge(listOf(dt1, dt2), {map1, map2 -> map1 + map2 }, ::doubleEquality)
        Assert.assertEquals(dt1, result)
    }

    @Test
    fun pRegress_binaryNodes_exponentialTreeSize(){
        val dbn = mapOf(
            X to DTDecision(Pair(X, 0),
                DTLeaf(Factor(listOf(X), listOf(0.1, 0.9))),
                DTLeaf(Factor(listOf(X), listOf(0.9, 0.1)))
            ),
            Y to DTDecision(Pair(Y, 0),
                DTLeaf(Factor(listOf(Y), listOf(0.2, 0.8))),
                DTLeaf(Factor(listOf(Y), listOf(0.8, 0.2)))
            ),
            Z to DTDecision(Pair(Z, 0),
                DTLeaf(Factor(listOf(Z), listOf(0.3, 0.7))),
                DTLeaf(Factor(listOf(Z), listOf(0.7, 0.3)))
            )
        )

        val valTree = DTDecision(Pair(X, 0),
            DTDecision(Pair(Y, 0),
                DTLeaf(1.0),
                DTLeaf(2.0)
            ),
            DTDecision(Pair(Z, 0),
                DTLeaf(3.0),
                DTLeaf(4.0)
            )
        )
        val result = pRegress(valTree, dbn)
        Assert.assertEquals(8, numLeaves(result))
    }

    @Test
    fun pRegress_binaryNodesLopsidedValTree_exponentialTreeSize(){
        val dbn = mapOf(
            X to DTDecision(Pair(X, 0),
                DTLeaf(Factor(listOf(X), listOf(0.1, 0.9))),
                DTLeaf(Factor(listOf(X), listOf(0.9, 0.1)))
            ),
            Y to DTDecision(Pair(Y, 0),
                DTLeaf(Factor(listOf(Y), listOf(0.2, 0.8))),
                DTLeaf(Factor(listOf(Y), listOf(0.8, 0.2)))
            ),
            Z to DTDecision(Pair(Z, 0),
                DTLeaf(Factor(listOf(Z), listOf(0.3, 0.7))),
                DTLeaf(Factor(listOf(Z), listOf(0.7, 0.3)))
            ),
            W to DTDecision(Pair(W, 0),
                DTLeaf(Factor(listOf(W), listOf(0.3, 0.7))),
                DTLeaf(Factor(listOf(W), listOf(0.7, 0.3)))
            )
        )

        val valTree = DTDecision(Pair(X, 0),
            DTDecision(Pair(Y, 0),
                DTDecision(Pair(Z, 0),
                    DTDecision(Pair(W, 0),
                        DTLeaf(3.0),
                        DTLeaf(6.0)
                    ),
                    DTLeaf(4.0)
                ),
                DTLeaf(2.0)
            ),
            DTLeaf(5.0)
        )
        val result = pRegress(valTree, dbn)
        Assert.assertEquals(16, numLeaves(result))
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
        val expectedFVTree = DTDecision(Pair(Z, 0),
            DTDecision(Pair(Y, 0),
                DTLeaf(0.0),
                DTLeaf(9.0)
            ),
            DTLeaf(10.0)
        )
        Assert.assertTrue(checkEquality(expectedFVTree, fvTree, ::doubleEquality))
    }

    @Test
    fun regress_initialReward_discountedValueTree(){
        val expected= DTDecision(Pair(Z,0),
            DTDecision(Pair(Y,0),
                DTLeaf(0.0),
                DTLeaf(8.1)
            ),
            DTLeaf(19.0)
        )
        val result = regress(rewardTree, rewardTree, dbn)
        Assert.assertTrue(checkEquality(expected, result, ::doubleEquality))
    }

    @Test
    fun pRegress_valueAfter1step_pTree(){
        val expected = DTDecision(Pair(Z, 0),
            DTDecision(Pair(Y, 0),
                DTDecision(Pair(X, 0),
                    DTLeaf(mapOf(Z to detFactor(Z, 0), Y to detFactor(Y, 0))),
                    DTLeaf(mapOf(Z to detFactor(Z, 0), Y to Factor(listOf(Y), listOf(0.1, 0.9))))
                ),
                DTLeaf(mapOf(Z to Factor(listOf(Z), listOf(0.1, 0.9)), Y to detFactor(Y, 1)))
            ),
            DTLeaf(mapOf(Z to detFactor(Z, 1)))
        )
        val result = pRegress(value1Step, dbn)

        Assert.assertTrue(checkEquality(expected, result, ::doubleEquality))
    }

    @Test
    fun futureValue_valueAfter1Step_FVTree(){
        val expected = DTDecision(Pair(Z,0),
            DTDecision(Pair(Y,0),
                DTDecision(Pair(X,0),
                    DTLeaf(0.0),
                    DTLeaf(7.29)
                ),
                DTLeaf(17.91)
            ),
            DTLeaf(19.0)
        )
        val pTree = pRegress(value1Step, dbn)
        val result = undiscountedFutureValue(value1Step, pTree)
        Assert.assertTrue(checkEquality(expected, result, ::doubleEquality))
    }

    @Test
    fun regress_valueAfter1Step_discountedValueTree(){
        val expected = DTDecision(Pair(Z,0),
            DTDecision(Pair(Y,0),
                DTDecision(Pair(X,0),
                    DTLeaf(0.0),
                    DTLeaf(6.561)
                ),
                DTLeaf(16.119)
            ),
            DTLeaf(27.1)
        )
        val result = regress(value1Step, rewardTree, dbn)

        Assert.assertTrue(checkEquality(expected, result, ::doubleEquality))
    }

    @Test
    fun simplify_duplicateBranch_returnTopMost(){
        val dt = DTDecision(Pair(X, 0),
            DTDecision(Pair(X, 0),
                DTLeaf(3.0),
                DTLeaf(4.0)
            ),
            DTLeaf(4.0)
        )

        val expected = DTDecision(Pair(X, 0),
            DTLeaf(3.0),
            DTLeaf(4.0)
        )

        val result = simplify(dt, ::doubleEquality)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun simplify_passBranchInverseTest_stillSimplifies(){
        val dt = DTDecision(Pair(X, 0),
            DTDecision(Pair(X, 1),
                DTLeaf(4.0),
                DTLeaf(3.0)
            ),
            DTLeaf(4.0)
        )

        val expected = DTDecision(Pair(X, 0),
            DTLeaf(3.0),
            DTLeaf(4.0)
        )

        val result = simplify(dt, ::doubleEquality)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun simplify_FailBranchInverseTest_stillSimplifies(){
        val dt = DTDecision(Pair(X, 0),
            DTLeaf(3.0),
            DTDecision(Pair(X, 1),
                DTLeaf(4.0),
                DTLeaf(3.0)
            )
        )

        val expected = DTDecision(Pair(X, 0),
            DTLeaf(3.0),
            DTLeaf(4.0)
        )

        val result = simplify(dt, ::doubleEquality)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun simplify_tripleRV_dontSimplifyFail(){
        val T = RandomVariable("T", listOf("t1", "t2", "t3"))
        val dt = DTDecision(Pair(T, 0),
            DTLeaf(3.0),
            DTDecision(Pair(T, 1),
                DTLeaf(4.0),
                DTLeaf(5.0)
            )
        )

        val expected = DTDecision(Pair(T, 0),
            DTLeaf(3.0),
            DTDecision(Pair(T, 1),
                DTLeaf(4.0),
                DTLeaf(5.0)
            )
        )

        val result = simplify(dt, ::doubleEquality)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun simplify_tripleRV_stillSimplifyPass(){
        val T = RandomVariable("T", listOf("t1", "t2", "t3"))
        val dt = DTDecision(Pair(T, 0),
            DTDecision(Pair(T, 1),
                DTLeaf(4.0),
                DTLeaf(3.0)
            ),
            DTLeaf(4.0)
        )

        val expected = DTDecision(Pair(T, 0),
            DTLeaf(3.0),
            DTLeaf(4.0)
        )

        val result = simplify(dt, ::doubleEquality)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun simplify_tripleRV_simplifyIfTestsUsedUp(){
        val T = RandomVariable("T", listOf("t1", "t2", "t3"))
        val dt = DTDecision(Pair(T, 0),
            DTLeaf(3.0),
            DTDecision(Pair(T, 1),
                DTLeaf(4.0),
                DTDecision(Pair(T, 2),
                    DTLeaf(5.0),
                    DTLeaf(4.0)
                )
            )
        )

        val expected = DTDecision(Pair(T, 0),
            DTLeaf(3.0),
            DTDecision(Pair(T, 1),
                DTLeaf(4.0),
                DTLeaf(5.0)
            )
        )

        val result = simplify(dt, ::doubleEquality)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun choosePolicy_SameLabelsSameRV_mergeIntoOne(){
        val qTrees = mapOf<Action, DecisionTree<Double>>(
            "A1" to DTDecision(Pair(X,0),
                DTLeaf(3.0),
                DTLeaf(5.0)
            ),
            "A2" to DTDecision(Pair(X, 0),
                DTLeaf(6.0),
                DTLeaf(4.0)
            )
        )

        val expected = DTDecision(Pair(X, 0),
            DTLeaf("A2"),
            DTLeaf("A1")
        )

        val result = choosePolicy(qTrees.mapValues { toRanged(it.value) }, listOf(X))
        Assert.assertEquals(expected, result)
    }

    @Test
    fun choosePolicy_DiffLabelsSameRV_mergeIntoOne(){
        val qTrees = mapOf<Action, DecisionTree<Double>>(
            "A1" to DTDecision(Pair(X,0),
                DTLeaf(3.0),
                DTLeaf(5.0)
            ),
            "A2" to DTDecision(Pair(X, 1),
                DTLeaf(4.0),
                DTLeaf(6.0)
            )
        )

        val expected = DTDecision(Pair(X, 0),
            DTLeaf("A2"),
            DTLeaf("A1")
        )

        val result = choosePolicy(qTrees.mapValues { toRanged(it.value) }, listOf(X))
        Assert.assertEquals(expected, result)
    }

    @Test
    fun merge_SameLabelsSameRV_mergeCorrectly(){
        val qTrees = mapOf<Action, DecisionTree<Double>>(
            "A1" to DTDecision(Pair(X,0),
                DTLeaf(3.0),
                DTLeaf(5.0)
            ),
            "A2" to DTDecision(Pair(X, 0),
                DTLeaf(6.0),
                DTLeaf(4.0)
            )
        )

        val expected = DTDecision(Pair(X, 0),
            DTLeaf(6.0),
            DTLeaf(5.0)
        )

        val result = merge(qTrees.values, ::maxOf, ::doubleEquality)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun merge_DifferentLabelsSameRV_mergeCorrectly(){
        val qTrees = mapOf<Action, DecisionTree<Double>>(
            "A1" to DTDecision(Pair(X,0),
                DTLeaf(3.0),
                DTLeaf(5.0)
            ),
            "A2" to DTDecision(Pair(X, 1),
                DTLeaf(4.0),
                DTLeaf(6.0)
            )
        )

        val expected = DTDecision(Pair(X, 0),
            DTLeaf(6.0),
            DTLeaf(5.0)
        )

        val result = merge(qTrees.values, ::maxOf, ::doubleEquality)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun merge_TripleValRV_mergeCorrectly(){
        val qTrees = mapOf<Action, DecisionTree<Double>>(
            "A1" to DTDecision(Pair(T,0),
                DTLeaf(3.0),
                DTLeaf(5.0)
            ),
            "A2" to DTDecision(Pair(T, 1),
                DTLeaf(4.0),
                DTLeaf(6.0)
            )
        )

        val expected = DTDecision(Pair(T, 0),
            DTLeaf(6.0),
            DTDecision(Pair(T, 1),
                DTLeaf(5.0),
                DTLeaf(6.0)
            )
        )

        val result = merge(qTrees.values, ::maxOf, ::doubleEquality)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun choosePolicy_TripleValRV_mergeCorrectly(){
        val qTrees = mapOf<Action, DecisionTree<Double>>(
            "A1" to DTDecision(Pair(T,0),
                DTLeaf(3.0),
                DTLeaf(5.0)
            ),
            "A2" to DTDecision(Pair(T, 1),
                DTLeaf(4.0),
                DTLeaf(6.0)
            )
        )

        val expected = DTDecision(Pair(T, 0),
            DTLeaf("A2"),
            DTDecision(Pair(T, 1),
                DTLeaf("A1"),
                DTLeaf("A2")
            )
        )

        val result = choosePolicy(qTrees.mapValues { toRanged(it.value) }, listOf(T))
        Assert.assertEquals(expected, result)
    }

}
