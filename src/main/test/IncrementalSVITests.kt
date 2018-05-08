import org.junit.Assert
import org.junit.Test

class IncrementalSVITests{
    @Test
    fun convertFromITI_emptyLeaf_Reward0(){
        val rewardTree = ITILeaf<Trial, Double>(emptyMap(), emptyMap(), ArrayList(), HashMap(), false)
        val result = convertFromITI(rewardTree)
        val expected = DTLeaf(0.0)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun convertFromITI_leafSingleExample_LeafWithReward(){
        val rewardTree = ITILeaf<Trial,Double>(emptyMap(), emptyMap(), mutableListOf(Trial(mapOf(A to 0), 5.0)), HashMap(), false)
        val result = convertFromITI(rewardTree)
        val expected = DTLeaf(5.0)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun convertFromITI_leafMutliExample_returnLeafWithAverageReward(){
        val rewardTree = ITILeaf<Trial,Double>(emptyMap(), emptyMap(), mutableListOf(Trial(mapOf(A to 0), 5.0), Trial(mapOf(A to 0), 10.0)), HashMap(), false)
        val result = convertFromITI(rewardTree)
        val expected = DTLeaf(7.5)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun convertFromITI_decision_returnDTWithCorrespondingRewards(){
        val rewardTree = ITIDecision<Trial, Double>(emptyMap(), emptyMap(), Pair(A, 0),
            ITILeaf(emptyMap(), emptyMap(), mutableListOf(Trial(mapOf(A to 0), 10.0)), HashMap(), false),
            ITILeaf(emptyMap(), emptyMap(), mutableListOf(Trial(mapOf(A to 1), 5.0)), HashMap(), false),
            false)
        val result = convertFromITI(rewardTree)
        val expected = DTDecision(Pair(A, 0),
            DTLeaf(10.0),
            DTLeaf(5.0)
        )

        Assert.assertEquals(expected, result)
    }

    @Test
    fun convertToCPT_leaf_leafWithCorrespondingFactorIncludingPrior(){
        val trials = listOf(
            SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 0), 0.0),
            SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 0), 0.0),
            SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 0), 0.0),
            SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 1), 0.0),
            SequentialTrial(mapOf(A to 1), "A1", mapOf(A to 1), 0.0)
        )

        val priorParams = DTLeaf(Factor(listOf(A), listOf(0.5, 0.5)))


        val probConfig = probTreeConfig(A, setOf(A), 5.0, 0.7)
        val probTree = makeLeaf(trials, probConfig)

        val result = convertToCPT(A, probTree, priorParams, 5.0)
        val expected = DTLeaf(Factor(listOf(A), listOf((3 + 2.5) / 10.0, (2 + 2.5) / 10.0)))
        Assert.assertEquals(expected, result)
    }

    @Test
    fun convertToCPT_decisionTree_correspondingDecisionTree(){
        val trials = listOf(
            SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 0), 0.0),
            SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 0), 0.0),
            SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 0), 0.0),
            SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 1), 0.0),
            SequentialTrial(mapOf(A to 1), "A1", mapOf(A to 1), 0.0)
        )

        val priorParams = DTDecision(Pair(A, 0),
            DTLeaf(Factor(listOf(A), listOf(0.25, 0.25))),
            DTLeaf(Factor(listOf(A), listOf(0.25, 0.25)))
        )


        val probConfig = probTreeConfig(A, setOf(A), 5.0, 0.7)
        val probTree = ITIDecision(
            emptyMap(),
            emptyMap(),
            Pair(A, 0),
            makeLeaf(trials.subList(0, 4), probConfig, branchLabel = mapOf(Pair(A,0) to true)),
            makeLeaf(trials.subList(4, 5), probConfig, branchLabel = mapOf(Pair(A,0) to false)),
            false
        )


        val result = convertToCPT(A, probTree, priorParams, 5.0)
        val expected = DTDecision(Pair(A, 0),
            DTLeaf(Factor(listOf(A), listOf((3.0 + 1.25) / 6.5, (1.0 + 1.25) / 6.5))),
            DTLeaf(Factor(listOf(A), listOf((0.0 + 1.25) / 3.5, (1.0 + 1.25) / 3.5)))
        )

        Assert.assertEquals(expected, result)
    }

    @Test
    fun jointQuery_fullAssignment_matchLeafAndReturn(){
        val parentAssignment = mapOf(Pair(A,0) to true)
        val jointDT = DTDecision(Pair(A, 0),
            DTLeaf(Factor(listOf(A), listOf(0.05, 0.45))),
            DTLeaf(Factor(listOf(A), listOf(0.2, 0.3)))
        )

        val result = jointQuery(parentAssignment, jointDT)
        Assert.assertEquals(Factor(listOf(A), listOf(0.05, 0.45)), result)
    }

    @Test
    fun jointQuery_emptyAssignment_addAllLeaves(){
        val parentAssignment = emptyMap<RVTest, Boolean>()
        val jointDT = DTDecision(Pair(A, 0),
            DTLeaf(Factor(listOf(A), listOf(0.05, 0.45))),
            DTLeaf(Factor(listOf(A), listOf(0.2, 0.3)))
        )

        val result = jointQuery(parentAssignment, jointDT)
        Assert.assertEquals(Factor(listOf(A), listOf(0.25, 0.75)), result)
    }

    @Test
    fun jointQuery_partialPAssignment_addSpecificLeaves(){
        val parentAssignment = mapOf(Pair(A,0) to true)
        val jointDT = DTDecision(Pair(B, 0),
            DTDecision(Pair(A, 0),
                DTLeaf(Factor(listOf(A), listOf(0.25, 0.05))),
                DTLeaf(Factor(listOf(A), listOf(0.1, 0.1)))
            ),
            DTLeaf(Factor(listOf(A), listOf(0.2, 0.3)))
        )

        val result = jointQuery(parentAssignment, jointDT)
        Assert.assertEquals(Factor(listOf(A), listOf(0.35, 0.2)), result)
    }

    @Test
    fun convertJointProbTree_Leaf_ScaleCounts(){
        val priorJointDT = DTLeaf(Factor(listOf(A), listOf(0.4, 0.6)))
        val probConfig = probTreeConfig(A, setOf(A), 5.0, 0.7)
        val binaryPTree = makeLeaf(listOf(
            SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 1), 0.0),
            SequentialTrial(mapOf(A to 1), "A1", mapOf(A to 0), 0.0)
        ), probConfig)

        val result = convertToJointProbTree(A, binaryPTree, priorJointDT, 5.0)
        val expected = DTLeaf(Factor(listOf(A), listOf((1.0 + 2.0) / 7.0, (1.0 + 3.0) / 7.0)))
        Assert.assertEquals(expected, result)
    }

    @Test
    fun convertJointProbTree_Decision_ScaleByTotalCountsInWholeTree(){
        val trials = listOf(
            SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 1), 0.0),
            SequentialTrial(mapOf(A to 1), "A1", mapOf(A to 0), 0.0)
        )

        val priorJointDT = DTLeaf(Factor(listOf(A), listOf(0.4, 0.6)))
        val probConfig = probTreeConfig(A, setOf(A), 5.0, 0.7)
        val binaryPTree = ITIDecision(
            emptyMap(),
            createStats(setOf(A), trials, { test, trial -> trial.prevState[test.first] == test.second }, { trial -> trial.currentState[A]!! }),
            Pair(A, 0),
            makeLeaf(listOf(SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 1), 0.0)), probConfig, mapOf(Pair(A, 0) to true)),
            makeLeaf(listOf(SequentialTrial(mapOf(A to 1), "A1", mapOf(A to 0), 0.0)), probConfig, mapOf(Pair(A, 0) to false)),
            false
        )

        val result = convertToJointProbTree(A, binaryPTree, priorJointDT, 5.0)
        val expected = DTDecision(Pair(A, 0),
            DTLeaf(Factor(listOf(A), listOf((0.0 + 0.4 * 2.5) / 7.0, (1.0 + 0.6 * 2.5) / 7.0))),
            DTLeaf(Factor(listOf(A), listOf((1.0 + 0.4 * 2.5) / 7.0, (0.0 + 0.6 * 2.5) / 7.0)))
        )


        Assert.assertEquals(expected, result)
    }

    @Test
    fun incrementalSVI_singleRewardStepBoutillier_takesOneStep(){
        val rewardTree = DTDecision(Pair(A, 0),
            DTLeaf(10.0),
            DTLeaf(0.0)
        )

        val valueTree = DTDecision(Pair(A, 0),
            DTLeaf(10.0),
            DTLeaf(0.0)
        )

        val actionDBNs = mapOf(
            "A1" to mapOf(A to DTLeaf(Factor(listOf(A), listOf(1.0, 0.0)))),
            "A2" to mapOf(A to DTLeaf(Factor(listOf(A), listOf(0.0, 1.0))))
        )

        val (valTreeResult, qTreesResult) = incrementalSVI(rewardTree, toRanged(valueTree), actionDBNs, listOf(A), 0.1)
        val expectedValTree = DTDecision(Pair(A, 0),
            DTLeaf(Range(19.0, 19.0)),
            DTLeaf(Range(9.0, 9.0))
        )
        val expectedQTrees = mapOf(
            "A1" to DTDecision(Pair(A, 0),
                DTLeaf(Range(19.0, 19.0)),
                DTLeaf(Range(9.0, 9.0))
            ),
            "A2" to DTDecision(Pair(A, 0),
                DTLeaf(Range(10.0, 10.0)),
                DTLeaf(Range(0.0, 0.0))
            ))
        Assert.assertEquals(expectedValTree, valTreeResult)
        Assert.assertEquals(expectedQTrees, qTreesResult)
    }

    @Test
    fun prune_leaf_returnSameLeaf(){
        val dt = DTLeaf(Range(5.0, 5.0))
        val result = prune(dt, 0.1)
        Assert.assertEquals(dt, result)
    }

    @Test
    fun prune_singleDec_rangeIsFarApart_returnOriginal(){
        val dt = DTDecision(Pair(A, 0),
            DTLeaf(Range(100.0, 100.0)),
            DTLeaf(Range(50.0, 50.0))
        )
        val result = prune(dt, 0.1)
        Assert.assertEquals(dt, result)
    }

    @Test
    fun prune_singleDec_rangeIsClose_returnMergedLeaf(){
        val dt = DTDecision(Pair(A, 0),
            DTLeaf(10.0),
            DTLeaf(10.1)
        )
        val result = prune(dt, 0.2)
        val expected = DTLeaf(Range(10.0, 10.1))
        Assert.assertEquals(expected, result)
    }

    @Test
    fun prune_recursiveDecisionAllCloseRange_returnLeaf(){
        val dt = DTDecision(Pair(A, 0),
            DTLeaf(0.0),
            DTDecision(Pair(B, 0),
                DTLeaf(1.0),
                DTLeaf(3.0)
            )
        )
        val result = prune(dt, 5.0)
        val expected = DTLeaf(Range(0.0, 3.0))

        Assert.assertEquals(expected, result)
    }

    @Test
    fun prune_recursiveDecOneStepCollapsableButTotalRangeAbove_OnlyDoSingleCollapse(){
        val dt = DTDecision(Pair(A, 0),
            DTLeaf(0.0),
            DTDecision(Pair(B, 0),
                DTLeaf(2.0),
                DTLeaf(6.0)
            )
        )
        val result = prune(dt, 5.0)
        val expected = DTDecision(Pair(A, 0),
            DTLeaf(Range(0.0, 0.0)),
            DTLeaf(Range(2.0, 6.0))
        )

        Assert.assertEquals(expected, result)
    }

    @Test
    fun changeAllowedVocab_SameVocab_SameLeaf(){
        val probConfig = probTreeConfig(A, setOf(A, B), 5.0, 0.7)
        val dtNode = makeLeaf(listOf(SequentialTrial(mapOf(A to 0, B to 0), "A1", mapOf(A to 0, B to 0), 5.0)), probConfig)
        val tree = Pair(dtNode, probConfig)
        val result = changeAllowedVocab(tree, setOf(A, B))
        Assert.assertEquals(tree, result)
    }

    @Test
    fun changeAllowedVocab_SmallerVocab_FilterTests(){
        val probConfig = probTreeConfig(A, setOf(A, B), 5.0, 0.7)
        val dtNode = makeLeaf(listOf(SequentialTrial(mapOf(A to 0, B to 0), "A1", mapOf(A to 0, B to 0), 5.0)), probConfig)
        val tree = Pair(dtNode, probConfig)
        val result = changeAllowedVocab(tree, setOf(A))

        val expectedConfig = probTreeConfig(A, setOf(A), 5.0, 0.7)
        val expectedTree = makeLeaf(listOf(SequentialTrial(mapOf(A to 0, B to 0), "A1", mapOf(A to 0, B to 0), 5.0)), expectedConfig, stale = true)
        Assert.assertEquals(expectedTree, result.first)
        Assert.assertEquals(expectedConfig.vocab, result.second.vocab)
    }

    @Test
    fun changeAllowedVocab_LargerVocab_RecountAndExpandTests(){
        val probConfig = probTreeConfig(A, setOf(A, B), 5.0, 0.7)
        val dtNode = makeLeaf(listOf(SequentialTrial(mapOf(A to 1, B to 1, C to 1), "A1", mapOf(A to 0, B to 0), 5.0)), probConfig)
        val tree = Pair(dtNode, probConfig)
        val result = changeAllowedVocab(tree, setOf(A, B, C))

        val expectedConfig = probTreeConfig(A, setOf(A, B, C), 5.0, 0.7)
        val expectedTree = makeLeaf(listOf(SequentialTrial(mapOf(A to 1, B to 1, C to 1), "A1", mapOf(A to 0, B to 0), 5.0)), expectedConfig, stale = true)
        Assert.assertEquals(expectedTree, result.first)
        Assert.assertEquals(expectedConfig.vocab, result.second.vocab)
    }

    @Test
    fun changeAllowedVocab_BranchingDecision_ApplyRecursively(){
        val probConfig = probTreeConfig(A, setOf(A, B), 5.0, 0.7)
        val trials = listOf(
            SequentialTrial(mapOf(A to 1, B to 1, C to 1), "A1", mapOf(A to 0, B to 0, C to 0), 5.0)
        )
        val dtNode = ITIDecision(emptyMap(),
            createStats(trials, probConfig),
            Pair(A, 1),
            makeLeaf(trials, probConfig, mapOf(Pair(A,1) to true)),
            makeLeaf(emptyList(), probConfig, mapOf(Pair(A,1) to false)),
            false
        )
        val result = changeAllowedVocab(Pair(dtNode, probConfig), setOf(A, B, C))
        val expectedConfig = probTreeConfig(A, setOf(A, B, C), 5.0, 0.7)
        val expectedDT = ITIDecision(emptyMap(),
            createStats(trials, expectedConfig),
            Pair(A, 1),
            makeLeaf(trials, expectedConfig, mapOf(Pair(A,1) to true), stale = true),
            makeLeaf(emptyList(), expectedConfig, mapOf(Pair(A,1) to false), stale = true),
            true
        )

        Assert.assertEquals(expectedConfig.vocab, result.second.vocab)
        Assert.assertEquals(expectedDT, result.first)
    }
}
