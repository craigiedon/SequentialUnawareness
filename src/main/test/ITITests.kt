import org.junit.*

fun <T, C> makeLeaf(examples : List<T>, config : ITIUpdateConfig<T, C>, branchLabel : Map<RVTest, Boolean> = emptyMap(), stale : Boolean = false) : ITILeaf<T,C>{
    return ITILeaf(branchLabel, createStats(examples, config), examples, classCounts(examples, config.exampleToClass), stale)
}

class ITITests{
    @Test
    fun incUpdate_emptyTree_singleLeaf(){
        val vocab = setOf(A, B, C)
        val rConfig = rewardTreeConfig(vocab)

        val trial = Trial(mapOf(A to 0, B to 0, C to 0), 10.0)
        val trials = listOf(trial)

        val startEmpty = emptyNode(rConfig)

        val expected = RLeaf(emptyMap(), createStats(trials, rConfig), trials, classCounts(trials, rConfig.exampleToClass), false)
        val result = incrementalUpdate(startEmpty, listOf(trial), rConfig)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun incUpdate_identicalTrial_keepLeaf(){
        val vocab = setOf(A, B, C)
        val rConfig = rewardTreeConfig(vocab)
        val trial = Trial(mapOf(A to 0, B to 0, C to 0), 10.0)
        val identicalTrial = trial.copy()

        val initialTrials = listOf(trial)
        val finalTrials = listOf(trial, identicalTrial)

        val initialLeaf = RLeaf(emptyMap(), createStats(initialTrials, rConfig), initialTrials, classCounts(initialTrials, rConfig.exampleToClass), false)

        val expected = RLeaf(emptyMap(), createStats(finalTrials, rConfig), finalTrials, classCounts(finalTrials, rConfig.exampleToClass), false)
        val result = incrementalUpdate(initialLeaf, listOf(identicalTrial), rConfig)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun incUpdate_diffAssignmentSameReward_keepLeaf(){
        val vocab = setOf(A, B, C)
        val rConfig = rewardTreeConfig(vocab)
        val trial = Trial(mapOf(A to 0, B to 0, C to 0), 10.0)
        val diffAssignmentSameReward = Trial(mapOf(A to 0, B to 0, C to 1), 10.0)

        val initialLeaf = makeLeaf(listOf(trial), rConfig)

        val expected = makeLeaf(listOf(trial, diffAssignmentSameReward), rConfig)
        val result = incrementalUpdate(initialLeaf, listOf(diffAssignmentSameReward), rConfig)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun incUpdate_diffAssignmentDiffReward_splitIntoDecision(){
        val vocab = setOf(A, B, C)
        val rConfig = rewardTreeConfig(vocab)
        val trial = Trial(mapOf(A to 0, B to 0, C to 0), 10.0)
        val diffReward = Trial(mapOf(A to 0, B to 0, C to 1), 15.0)

        val initialLeaf = makeLeaf(listOf(trial), rConfig)

        val expectedPassBranch = makeLeaf(listOf(trial), rConfig, mapOf(Pair(C,0) to true))
        val expectedFailBranch = makeLeaf(listOf(diffReward), rConfig, mapOf(Pair(C, 0) to false))

        val result = incrementalUpdate(initialLeaf, listOf(diffReward), rConfig) as RDecision
        Assert.assertEquals(Pair(C, 0), result.currentTest)
        Assert.assertEquals(expectedPassBranch, result.passBranch)
        Assert.assertEquals(expectedFailBranch, result.failBranch)
    }

    @Test
    fun addExamples_missingAssignment_putInFailBranch(){
        val vocab = setOf(A, B, C)
        val rConfig = rewardTreeConfig(vocab)

        val dt = RDecision(
            emptyMap(),
            createStats(emptyList(), rConfig),
            Pair(B, 0),
            makeLeaf(emptyList(), rConfig, mapOf(Pair(B,0) to true)),
            RDecision(
                mapOf(Pair(B,0) to false),
                createStats(emptyList(), rConfig),
                Pair(C, 1),
                makeLeaf(emptyList(), rConfig, mapOf(Pair(B,0) to false, Pair(C, 1) to true)),
                makeLeaf(emptyList(), rConfig, mapOf(Pair(B, 0) to false, Pair(C, 1) to false)),
                false
            ),
            false
        )

        val missingAssignments = Trial(mapOf(A to 0), 10.0)

        val expected = RDecision(
            emptyMap(),
            createStats(listOf(missingAssignments), rConfig),
            Pair(B, 0),
            makeLeaf(emptyList(), rConfig, mapOf(Pair(B, 0) to true)),
            RDecision(
                mapOf(Pair(B,0) to false),
                createStats(listOf(missingAssignments), rConfig),
                Pair(C, 1),
                makeLeaf(emptyList(), rConfig, mapOf(Pair(B,0) to false, Pair(C, 1) to true)),
                makeLeaf(listOf(missingAssignments), rConfig, mapOf(Pair(B,0) to false, Pair(C, 1) to false), true),
                stale = true),
            stale = true
        )

        val result = addExamples(dt, listOf(missingAssignments), rConfig)

        Assert.assertEquals(expected, result)
    }

    @Test
    fun tranpose_bothLeaves_replacementTestRVUnavailableButNewOneOpen(){
        val vocab = setOf(A,B,C)
        val rConfig = rewardTreeConfig(vocab)
        val trials = listOf(
            Trial(mapOf(A to 1, B to 1, C to 1), 15.0),
            Trial(mapOf(A to 0, B to 1, C to 0), 20.0)
        )

        val dt =
            RDecision(
                emptyMap(),
                createStats(trials, rConfig),
                Pair(C, 1),
                makeLeaf(listOf(trials[0]), rConfig, mapOf(Pair(C,0) to true)),
                makeLeaf(listOf(trials[1]), rConfig, mapOf(Pair(C, 0) to true)),
                stale = true
            )

        val transposedTree = transposeTree(dt, RVTest(A, 0), rConfig.vocab, rConfig.testChecker, rConfig.exampleToClass)

        val expected =
            RDecision(
                emptyMap(),
                createStats(trials, rConfig),
                Pair(A, 0),
                makeLeaf(listOf(trials[1]), rConfig, mapOf(Pair(A,0) to true), true),
                makeLeaf(listOf(trials[0]), rConfig, mapOf(Pair(A,0) to false), true),
                stale = true
            )

        Assert.assertEquals(expected, transposedTree)
    }

    @Test
    fun transpose_oneMatchingSubtreeOneLeaf_addLeafEntriesToSubtree(){
        val vocab = setOf(A,B,C)
        val rConfig = rewardTreeConfig(vocab)

        val topLevelStats = createStats(listOf(
            Trial(mapOf(A to 0, B to 0, C to 1), 10.0),
            Trial(mapOf(A to 1, B to 0, C to 1), 15.0),
            Trial(mapOf(A to 0, B to 0, C to 0), 10.0)),
            rConfig
        )

        val subTreeStats = createStats(listOf(
            Trial(mapOf(A to 0, B to 0, C to 1), 10.0),
            Trial(mapOf(A to 1, B to 0, C to 1), 15.0)),
            rConfig
        )

        val dt =
            RDecision(
                emptyMap(),
                topLevelStats,
                Pair(C, 1),
                RDecision(
                    mapOf(Pair(C,1) to true),
                    subTreeStats,
                    Pair(A, 0),
                    makeLeaf(listOf(Trial(mapOf(A to 0, B to 0, C to 1), 10.0)), rConfig, mapOf(Pair(C,1) to true, Pair(A,0) to true)),
                    makeLeaf(listOf(Trial(mapOf(A to 1, B to 0, C to 1), 15.0)), rConfig, mapOf(Pair(C,1) to true, Pair(A,0) to false)),
                    stale = false
                ),
                makeLeaf(listOf(Trial(mapOf(A to 0, B to 0, C to 0), 10.0)), rConfig, mapOf(Pair(C,1) to false)),
                stale = false
            )

        val transposedTree = transposeTree(dt, Pair(A, 0), rConfig.vocab, rConfig.testChecker, rConfig.exampleToClass)

        val expectedTree =
                RDecision(
                    emptyMap(),
                    topLevelStats,
                    Pair(A, 0),
                    makeLeaf(listOf(
                        Trial(mapOf(A to 0, B to 0, C to 1), 10.0),
                        Trial(mapOf(A to 0, B to 0, C to 0), 10.0)), rConfig, mapOf(Pair(A,0) to true), true),
                    makeLeaf(listOf(Trial(mapOf(A to 1, B to 0, C to 1), 15.0)), rConfig, mapOf(Pair(A,0) to false), false),
                    stale = true
                )
        Assert.assertEquals(expectedTree, transposedTree)
    }

    @Test
    fun transpose_bothMatchingDecisionNodes_swapAndMerge(){
        val vocab = setOf(A,B,C)
        val rConfig = rewardTreeConfig(vocab)

        val topLevelStats = createStats(listOf(
            Trial(mapOf(A to 0, B to 0, C to 1), 10.0),
            Trial(mapOf(A to 1, B to 0, C to 1), 15.0),
            Trial(mapOf(A to 0, B to 0, C to 0), 20.0),
            Trial(mapOf(A to 1, B to 0, C to 0), 25.0)),
            rConfig
        )

        val passStats = createStats(listOf(
            Trial(mapOf(A to 0, B to 0, C to 1), 10.0),
            Trial(mapOf(A to 1, B to 0, C to 1), 15.0)), rConfig)

        val failStats = createStats(listOf(
            Trial(mapOf(A to 0, B to 0, C to 0), 20.0),
            Trial(mapOf(A to 1, B to 0, C to 0), 25.0)
        ), rConfig)

        val dt =
            RDecision(
                emptyMap(),
                topLevelStats,
                Pair(C, 1),
                RDecision(
                    mapOf(Pair(C,1) to true),
                    passStats,
                    Pair(A, 0),
                    makeLeaf(listOf(Trial(mapOf(A to 0, B to 0, C to 1), 10.0)), rConfig, mapOf(Pair(C, 1) to true, Pair(A,0) to true)),
                    makeLeaf(listOf(Trial(mapOf(A to 1, B to 0, C to 1), 15.0)), rConfig, mapOf(Pair(C,1) to true, Pair(A,0) to false)),
                    stale = false
                ),
                RDecision(
                    mapOf(Pair(C,1) to false),
                    failStats,
                    Pair(A, 0),
                    makeLeaf(listOf(Trial(mapOf(A to 0, B to 0, C to 0), 20.0)), rConfig, mapOf(Pair(C,1) to false, Pair(A,0) to true)),
                    makeLeaf(listOf(Trial(mapOf(A to 1, B to 0, C to 0), 25.0)), rConfig, mapOf(Pair(C,1) to false, Pair(A, 0) to false)),
                    stale = false
                ),
                stale = true
            )

        val transposedTree = transposeTree(dt, Pair(A, 0), rConfig.vocab, rConfig.testChecker, rConfig.exampleToClass)

        val expectedPassStats = createStats(listOf(
            Trial(mapOf(A to 0, B to 0, C to 1), 10.0),
            Trial(mapOf(A to 0, B to 0, C to 0), 20.0)
        ), rConfig)

        val expectedFailStats = createStats(listOf(
            Trial(mapOf(A to 1, B to 0, C to 1), 15.0),
            Trial(mapOf(A to 1, B to 0, C to 0), 25.0)
        ), rConfig)

        val expectedTree =
            RDecision(
                emptyMap(),
                topLevelStats,
                Pair(A, 0),
                RDecision(
                    mapOf(Pair(A,0) to true),
                    expectedPassStats,
                    Pair(C, 1),
                    makeLeaf(listOf(Trial(mapOf(A to 0, B to 0, C to 1), 10.0)), rConfig, mapOf(Pair(A,0) to true, Pair(C,1) to true)),
                    makeLeaf(listOf(Trial(mapOf(A to 0, B to 0, C to 0), 20.0)), rConfig, mapOf(Pair(A,0) to true, Pair(C,1) to false)),
                    stale = true
                ),
                RDecision(
                    mapOf(Pair(A,0) to false),
                    expectedFailStats,
                    Pair(C, 1),
                    makeLeaf(listOf(Trial(mapOf(A to 1, B to 0, C to 1), 15.0)), rConfig, mapOf(Pair(A,0) to false, Pair(C,1) to true)),
                    makeLeaf(listOf(Trial(mapOf(A to 1, B to 0, C to 0), 25.0)), rConfig, mapOf(Pair(A,0) to false, Pair(C,1) to false)),
                    stale = true
                ),
                stale = true
            )
        nodesEqualDiagnosis(expectedTree, transposedTree)
        Assert.assertEquals(expectedTree, transposedTree)
    }

    @Test
    fun transpose_recursive_correctTree(){
        val vocab = setOf(A, B, C)
        val rConfig = rewardTreeConfig(vocab)
        val topLevelStats = createStats(listOf(
            Trial(mapOf(A to 0, B to 0, C to 0), 10.0),
            Trial(mapOf(A to 1, B to 1, C to 1), 15.0),
            Trial(mapOf(A to 0, B to 1, C to 0), 20.0)
        ), rConfig)

        val childStats = createStats(listOf(
            Trial(mapOf(A to 1, B to 1, C to 1), 15.0),
            Trial(mapOf(A to 0, B to 1, C to 0), 20.0)
        ), rConfig)

        val dt = RDecision(
            emptyMap(),
            topLevelStats,
            Pair(B, 0),
            makeLeaf(listOf(Trial(mapOf(A to 0, B to 0, C to 0), 10.0)), rConfig, mapOf(Pair(B,0) to true)),
            RDecision(
                mapOf(Pair(B,0) to false),
                childStats,
                Pair(C, 1),
                makeLeaf(listOf(Trial(mapOf(A to 1, B to 1, C to 1), 15.0)), rConfig, mapOf(Pair(B,0) to false, Pair(C,1) to true)),
                makeLeaf(listOf(Trial(mapOf(A to 0, B to 1, C to 0), 20.0)), rConfig, mapOf(Pair(B,0) to false, Pair(C,1) to false)),
                stale = true
            ),
            stale = true
        )

        val transposedTree = transposeTree(dt, RVTest(A, 0), rConfig.vocab, rConfig.testChecker, rConfig.exampleToClass)

        val expected = RDecision(
            emptyMap(),
            topLevelStats,
            Pair(A, 0),
            makeLeaf(listOf(Trial(mapOf(A to 0, B to 1, C to 0), 20.0), Trial(mapOf(A to 0, B to 0, C to 0), 10.0)), rConfig, mapOf(Pair(A,0) to true), true),
            makeLeaf(listOf(Trial(mapOf(A to 1, B to 1, C to 1), 15.0)), rConfig, mapOf(Pair(A,0) to false), true),
            stale = true
        )

        Assert.assertEquals(expected, transposedTree)
    }
}

fun <T, C> nodesEqualDiagnosis(dt1 : ITINode<T,C>, dt2 : ITINode<T,C>){
    if((dt1 is ITILeaf && dt2 is ITIDecision) || (dt1 is ITIDecision && dt2 is ITILeaf)){
        println("Node type mismatch at ${dt1.branchLabel}")
    }
    if(dt1.branchLabel != dt2.branchLabel){
        println("Branch label mismatch - dt1: ${dt1.branchLabel}, dt2: ${dt2.branchLabel}")
    }
    if(dt1.stale != dt2.stale){
        println("Stale mismatch at ${dt1.branchLabel}")
        println("DT1 : ${dt1.stale}")
        println("DT2: ${dt2.stale}")
    }
    if(dt1.testCandidates != dt2.testCandidates){
        println("Test candidate mismatch at ${dt1.branchLabel}")
        println("DT1: ${dt1.testCandidates}")
        println("DT2: ${dt2.testCandidates}")
    }

    if(dt1 is ITILeaf && dt2 is ITILeaf){
        if(dt1.examples != dt2.examples){
            println("Example mistmatch at ${dt1.branchLabel}")
            println("DT1: ${dt1.examples}")
            println("DT2: ${dt2.examples}")
        }
        if(dt1.counts != dt2.counts){
            println("Counts mistmatch at ${dt1.branchLabel}")
            println("DT1: ${dt1.counts}")
            println("DT2: ${dt2.counts}")
        }
    }
    else if(dt1 is ITIDecision && dt2 is ITIDecision){
        if(dt1.currentTest != dt2.currentTest){
            println("Current test mistmatch at ${dt1.branchLabel}")
            println("DT1: ${dt1.currentTest}")
            println("DT2: ${dt2.currentTest}")
        }
        nodesEqualDiagnosis(dt1.passBranch, dt2.passBranch)
        nodesEqualDiagnosis(dt1.failBranch, dt2.failBranch)
    }
}