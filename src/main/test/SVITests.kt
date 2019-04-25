import org.junit.*
import java.lang.IllegalArgumentException

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
    fun convertToCPT_deterministicParamPrior_deterministicDBN(){
        val itiDT = ITIDecision<SequentialTrial, Int>(emptyMap(), emptyMap(), Pair(T, 2),
            ITILeaf(mapOf(Pair(T, 2) to true), emptyMap(), emptyList(), mutableMapOf(2 to 4), false),
            ITIDecision(mapOf(Pair(T, 2) to false), emptyMap(), Pair(T, 1),
                ITILeaf(mapOf(Pair(T, 2) to false, Pair(T, 1) to true), emptyMap(), emptyList(), mutableMapOf(1 to 3), false),
                ITILeaf(mapOf(Pair(T, 2) to false, Pair(T, 1) to false), emptyMap(), emptyList(), mutableMapOf(0 to 4), false),
                false),
            false
        )

        val jointParamPrior = DTDecision(Pair(T, 2),
            DTLeaf(Factor(listOf(T), listOf(0.0,0.0,1.0 / 3.0))),
            DTDecision(Pair(T, 1),
                DTLeaf(Factor(listOf(T), listOf(0.0, 1.0 / 3.0, 0.0))),
                DTLeaf(Factor(listOf(T), listOf(1.0 / 3.0, 0.0, 0.0)))
            )
        )

        val expected = DTDecision(Pair(T, 2),
            DTLeaf(Factor(listOf(T), listOf(0.0,0.0,1.0))),
            DTDecision(Pair(T, 1),
                DTLeaf(Factor(listOf(T), listOf(0.0, 1.0, 0.0))),
                DTLeaf(Factor(listOf(T), listOf(1.0, 0.0, 0.0)))
            )
        )
        val result = convertToCPT(T, itiDT, jointParamPrior, 1.0)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun convertToCPT_deterministicParamPriorButDifferentStructure_deterministicDBN(){
        val itiDT = ITIDecision<SequentialTrial, Int>(emptyMap(), emptyMap(), Pair(T, 2),
            ITILeaf(mapOf(Pair(T, 2) to true), emptyMap(), emptyList(), mutableMapOf(2 to 4), false),
            ITIDecision(mapOf(Pair(T, 2) to false), emptyMap(), Pair(T, 1),
                ITILeaf(mapOf(Pair(T, 2) to false, Pair(T, 1) to true), emptyMap(), emptyList(), mutableMapOf(1 to 3), false),
                ITILeaf(mapOf(Pair(T, 2) to false, Pair(T, 1) to false), emptyMap(), emptyList(), mutableMapOf(0 to 4), false),
                false),
            false
        )

        val jointParamPrior = DTDecision(Pair(T, 1),
            DTLeaf(Factor(listOf(T), listOf(0.0, 1.0 / 3.0, 0.0))),
            DTDecision(Pair(T, 0),
                DTLeaf(Factor(listOf(T), listOf(1.0 / 3.0, 0.0, 0.0))),
                DTLeaf(Factor(listOf(T), listOf(0.0, 0.0, 1.0 / 3.0)))
            )
        )

        val expected = DTDecision(Pair(T, 2),
            DTLeaf(Factor(listOf(T), listOf(0.0,0.0,1.0))),
            DTDecision(Pair(T, 1),
                DTLeaf(Factor(listOf(T), listOf(0.0, 1.0, 0.0))),
                DTLeaf(Factor(listOf(T), listOf(1.0, 0.0, 0.0)))
            )
        )
        val result = convertToCPT(T, itiDT, jointParamPrior, 1.0)
        Assert.assertEquals(expected, result)
    }

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
    fun treeApendedToIdenticalTree_originalTree(){
        val mapifiedZ = fMap(zCPD) {mapOf(it.scope.first() to it)}
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
    fun pRegress_ternaryNode_failNotDeemedRelevant(){
        val dt = DTDecision(Pair(T, 0),
            DTLeaf(5),
            DTDecision(Pair(T, 1),
                DTLeaf(3),
                DTLeaf(4)
            )
        )

        val t0DetMap = mapOf(T to Factor(listOf(T), listOf(1.0, 0.0, 0.0)))

        val dbn = mapOf(T to DTLeaf(Factor(listOf(T), listOf(1.0, 0.0, 0.0))))
        val expected = DTLeaf(t0DetMap)
        val result = pRegress(dt, dbn)
        Assert.assertEquals(expected, result)
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
        val result = regress(rewardTree, rewardTree, dbn, 0.9)
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
        val result = regress(value1Step, rewardTree, dbn, 0.9)

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

    @Test
    fun terminalTree_coffeeStart(){
        val huc = RandomVariable("huc", listOf("yes", "no"))
        val terminalDescription = mapOf(huc to 0)
        val terminalTree = terminalTree(terminalDescription)
        val expectedTree = DTDecision(RVTest(huc, 0), DTLeaf(0.0), DTLeaf(Double.POSITIVE_INFINITY))

        Assert.assertEquals(expectedTree, terminalTree)
    }

    @Test
    fun terminalTree_factoryStart(){
        val connected = RandomVariable("connected", listOf("good", "bad", "f"))
        val terminalDescriptions = listOf(
            mapOf(connected to 0),
            mapOf(connected to 1)
        )
        val terminalTree = terminalTree(terminalDescriptions)
        val expectedTree = DTDecision(RVTest(connected, 0), DTLeaf(0.0), DTDecision(RVTest(connected, 1), DTLeaf(0.0), DTLeaf(Double.POSITIVE_INFINITY)))

        Assert.assertEquals(expectedTree, terminalTree)
    }

    @Test
    fun structuredValueIteration_noTerminalStates_valueEqualsReward(){
        val zKeepsValue = identityTransition(Z)
        val terminalDescriptions = emptyList<RVAssignment>()
        val (policyTree, valueTree) = structuredValueIteration(rewardTree, mapOf("A1" to mapOf(Z to zKeepsValue)), listOf(Z), terminalDescriptions, 0.9, 0.0)
        Assert.assertEquals(Range(0.0, 0.0), matchLeaf(valueTree, mapOf(Z to 0)).value)
        val z1Val = matchLeaf(valueTree, mapOf(Z to 1)).value.lower
        Assert.assertTrue(Math.abs(100.0 - z1Val) <= 0.1)
    }

    @Test
    fun structuredValueIteration_terminalStates_valueEqualsReward(){
        val zKeepsValue = identityTransition(Z)
        val terminalDescriptions = listOf(mapOf(Z to 1))
        val (policyTree, valueTree) = structuredValueIteration(rewardTree, mapOf("A1" to mapOf(Z to zKeepsValue)), listOf(Z), terminalDescriptions, 0.9, 0.0)
        Assert.assertEquals(Range(0.0, 0.0), matchLeaf(valueTree, mapOf(Z to 0)).value)
        Assert.assertEquals(Range(10.0, 10.0), matchLeaf(valueTree, mapOf(Z to 1)).value)
    }

    @Test
    fun structuredValueIteration_terminalStates_valueTreeExpanded(){
        val zKeepsValue = identityTransition(Z)
        val xKeepsValue = identityTransition(X)
        val rTree = DTDecision(RVTest(Z, 1), DTLeaf(10.0), DTLeaf(1.0))
        val terminalDescriptions = listOf(mapOf(X to 0))
        val (policyTree, valueTree) = structuredValueIteration(rTree, mapOf("A1" to mapOf(Z to zKeepsValue, X to xKeepsValue)), listOf(Z, X), terminalDescriptions, 0.9, 0.0)

        println(valueTree)
        Assert.assertEquals(4, numLeaves(valueTree))
        Assert.assertEquals(10.0, matchLeaf(valueTree, mapOf(X to 0, Z to 1)).value.lower, 0.1)
        Assert.assertEquals(1.0, matchLeaf(valueTree, mapOf(X to 0, Z to 0)).value.lower, 0.1)
        Assert.assertEquals(10.0, matchLeaf(valueTree, mapOf(X to 1, Z to 0)).value.lower, 0.1)
        Assert.assertEquals(100.0, matchLeaf(valueTree, mapOf(X to 1, Z to 1)).value.lower, 0.1)
    }

    @Test
    fun structuredValueIteration_terminalStates_valueTreePruned(){
        val zKeepsValue = DTDecision(RVTest(Z, 1),
            DTDecision(RVTest(X, 1),
                DTLeaf(detFactor(Z, 1)),
                DTLeaf(detFactor(Z, 0))
                ),
            DTDecision(RVTest(X, 1),
                DTLeaf(detFactor(Z, 0)),
                DTLeaf(detFactor(Z, 1))
            )
        )
        val xKeepsValue = identityTransition(X)
        val rTree = DTDecision(RVTest(Z, 1), DTLeaf(10.0), DTLeaf(1.0))
        val terminalDescriptions = listOf(mapOf(Z to 1))
        val (policyTree, valueTree) = structuredValueIteration(rTree, mapOf("A1" to mapOf(Z to zKeepsValue, X to xKeepsValue)), listOf(Z, X), terminalDescriptions, 0.9, 0.0)

        println(valueTree)
        Assert.assertEquals(3, numLeaves(valueTree))
    }

    @Test(expected=IllegalArgumentException::class)
    fun jointQuery_ConflictingAssignmentsImplicit_ZeroProbability(){
        val jointDT = unifStartIDTransJoint(A)
        val result = jointQuery(mapOf(Pair(A, 1) to true,Pair(A, 0) to true), jointDT)
    }

    @Test
    fun structuredValueIteration_terminalStateMentionsPrevious_pruneCorrectly(){
        val terminalTree : DecisionTree<Double> = terminalTree(mapOf(Z to 1))
        val fvTree : DecisionTree<Double> = DTDecision(RVTest(Z, 1),
            DTLeaf(5.0),
            DTDecision(RVTest(X, 1),
                DTLeaf(7.0),
                DTLeaf(10.0))
        )
        val valueTree : DecisionTree<Double> = append(fvTree, terminalTree, {x,y -> minOf(x,y)}, {x,y -> doubleEquality(x, y)})

        println(valueTree)
        Assert.assertEquals(3, numLeaves(valueTree))
    }

    /*
    @Test
    fun structuredValueIteration_coffeeExample_policyAndValuesCorrect(){
        val mdp = loadMDP("mdps/coffee.json")
        val (policy, rangedValTree) = structuredValueIteration(mdp.rewardTree, mdp.dbns, mdp.vocab.toList(), mdp.terminalDescriptions, mdp.discount, 0.0)
        var valTree = toRanged(mdp.rewardTree)
        for(i in 1..10){
            //saveToJson(policy, "logs/tests/policy")
            valTree = incrementalSVI(mdp.rewardTree, valTree, mdp.dbns, mdp.vocab.toList(), mdp.terminalDescriptions, mdp.discount, 0.0).first
            saveToJson(valTree, "logs/tests/valTree$i")
        }
        println("Done")
    }
    */
}
