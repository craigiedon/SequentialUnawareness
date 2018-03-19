import org.junit.*

class ITITests{
    @Test
    fun incUpdate_emptyTree_singleLeaf(){
        val vocab = setOf(A, B, C)
        val trial = Trial(mapOf(A to 0, B to 0, C to 0), 10.0)
        val trials = listOf(trial)

        val expected = RLeaf(vocab, vocab, createStats(vocab, trials), trials)
        val result = incrementalUpdate(RLeaf(vocab, vocab, emptyMap(), emptyList()), trial)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun incUpdate_identicalTrial_keepLeaf(){
        val vocab = setOf(A, B, C)
        val trial = Trial(mapOf(A to 0, B to 0, C to 0), 10.0)
        val identicalTrial = trial.copy()

        val expected = RLeaf(vocab, vocab, listOf(trial, identicalTrial))
        val result = incrementalUpdate(RLeaf(vocab, vocab, listOf(trial)), identicalTrial)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun incUpdate_diffAssignmentSameReward_keepLeaf(){
        val vocab = setOf(A, B, C)
        val trial = Trial(mapOf(A to 0, B to 0, C to 0), 10.0)
        val diffAssignmentSameReward = Trial(mapOf(A to 0, B to 0, C to 1), 10.0)

        val expected = RLeaf(vocab, vocab, listOf(trial, diffAssignmentSameReward))
        val result = incrementalUpdate(RLeaf(vocab, vocab, listOf(trial)), diffAssignmentSameReward)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun incUpdate_diffAssignmentDiffReward_splitIntoDecision(){
        val vocab = setOf(A, B, C)
        val trial = Trial(mapOf(A to 0, B to 0, C to 0), 10.0)
        val diffReward = Trial(mapOf(A to 0, B to 0, C to 1), 15.0)

        val result = incrementalUpdate(RLeaf(vocab, vocab, listOf(trial)), diffReward) as RDecision
        Assert.assertEquals(Pair(C, 0), result.bestTest)
        Assert.assertEquals(RLeaf(vocab, vocab - C, listOf(trial)), result.passBranch)
        Assert.assertEquals(RLeaf(vocab, vocab - C, listOf(diffReward)), result.failBranch)
    }

    @Test
    fun addExamples_missingAssignment_putInFailBranch(){
        val vocab = setOf(A, B, C)
        val dt = RDecision(
            vocab,
            vocab,
            emptyMap(),
            Pair(B, 0),
            RLeaf(vocab, vocab - B, emptyList()),
            RDecision(
                vocab,
                vocab - B,
                emptyMap(),
                Pair(C, 1),
                RLeaf(vocab, setOf(A), emptyList()),
                RLeaf(vocab, setOf(A), emptyList())
            )
        )

        val missingAssignments = Trial(mapOf(A to 0), 10.0)

        val expected = RDecision(
            vocab,
            vocab,
            emptyMap(),
            Pair(B, 0),
            RLeaf(vocab, vocab - B, emptyList()),
            RDecision(
                vocab,
                vocab - B,
                emptyMap(),
                Pair(C, 1),
                RLeaf(vocab, setOf(A), emptyList()),
                RLeaf(vocab, setOf(A), listOf(missingAssignments)),
                stale = true),
            stale = true
        )

        val result = addExamples(dt, listOf(missingAssignments))
        Assert.assertEquals(expected, result)
    }

    @Test
    fun tranpose_bothLeaves_replacementTestRVUnavailableButNewOneOpen(){
        val vocab = setOf(A,B,C)
        val dt =
            RDecision(
                vocab,
                vocab,
                emptyMap(),
                Pair(C, 1),
                RLeaf(vocab, setOf(A, B), listOf(Trial(mapOf(A to 1, B to 1, C to 1), 15.0))),
                RLeaf(vocab, setOf(A, B), listOf(Trial(mapOf(A to 0, B to 1, C to 0), 20.0)))
            )

        val transposedTree = transposeTree(dt, RVTest(A, 0))

        val expected =
            RDecision(
                vocab,
                vocab,
                emptyMap(),
                Pair(A, 0),
                RLeaf(vocab, setOf(B, C), listOf(Trial(mapOf(A to 0, B to 1, C to 0), 20.0))),
                RLeaf(vocab, setOf(B, C), listOf(Trial(mapOf(A to 1, B to 1, C to 1), 15.0))),
                stale = true
            )

        Assert.assertEquals(expected, transposedTree)
    }

    @Test
    fun transpose_oneMatchingSubtreeOneLeaf_addLeafEntriesToSubtree(){
        val vocab = setOf(A,B,C)

        val topLevelStats = createStats(vocab, listOf(
            Trial(mapOf(A to 0, B to 0, C to 1), 10.0),
            Trial(mapOf(A to 1, B to 0, C to 1), 15.0),
            Trial(mapOf(A to 0, B to 0, C to 0), 10.0))
        )

        val subTreeStats = createStats(vocab, listOf(
            Trial(mapOf(A to 0, B to 0, C to 1), 10.0),
            Trial(mapOf(A to 1, B to 0, C to 1), 15.0))
        )

        val dt =
            RDecision(
                vocab,
                vocab,
                topLevelStats,
                Pair(C, 1),
                RDecision(vocab, vocab - C,
                    subTreeStats,
                    Pair(A, 0),
                    RLeaf(vocab, setOf(B), listOf(Trial(mapOf(A to 0, B to 0, C to 1), 10.0))),
                    RLeaf(vocab, setOf(B), listOf(Trial(mapOf(A to 1, B to 0, C to 1), 15.0))),
                    stale = false
                ),
                RLeaf(vocab, setOf(A, B), listOf(Trial(mapOf(A to 0, B to 0, C to 0), 10.0)))
            )

        val transposedTree = transposeTree(dt, Pair(A, 0))

        val expectedTree =
                RDecision(vocab, vocab,
                    topLevelStats,
                    Pair(A, 0),
                    RLeaf(vocab, setOf(B, C), listOf(
                        Trial(mapOf(A to 0, B to 0, C to 1), 10.0),
                        Trial(mapOf(A to 0, B to 0, C to 0), 10.0))),
                    RLeaf(vocab, setOf(B, C), listOf(Trial(mapOf(A to 1, B to 0, C to 1), 15.0))),
                    stale = true
                )
        Assert.assertEquals(expectedTree, transposedTree)
    }

    @Test
    fun transpose_bothMatchingDecisionNodes_swapAndMerge(){
        val vocab = setOf(A,B,C)

        val topLevelStats = createStats(vocab, listOf(
            Trial(mapOf(A to 0, B to 0, C to 1), 10.0),
            Trial(mapOf(A to 1, B to 0, C to 1), 15.0),
            Trial(mapOf(A to 0, B to 0, C to 0), 20.0),
            Trial(mapOf(A to 1, B to 0, C to 0), 25.0))
        )

        val passStats = createStats(vocab, listOf(
            Trial(mapOf(A to 0, B to 0, C to 1), 10.0),
            Trial(mapOf(A to 1, B to 0, C to 1), 15.0)
        ))

        val failStats = createStats(vocab, listOf(
            Trial(mapOf(A to 0, B to 0, C to 0), 20.0),
            Trial(mapOf(A to 1, B to 0, C to 0), 25.0)
        ))

        val dt =
            RDecision(
                vocab,
                vocab,
                topLevelStats,
                Pair(C, 1),
                RDecision(vocab, vocab - C,
                    passStats,
                    Pair(A, 0),
                    RLeaf(vocab, setOf(B), listOf(Trial(mapOf(A to 0, B to 0, C to 1), 10.0))),
                    RLeaf(vocab, setOf(B), listOf(Trial(mapOf(A to 1, B to 0, C to 1), 15.0))),
                    stale = false
                ),
                RDecision(vocab, vocab - C,
                    failStats,
                    Pair(A, 0),
                    RLeaf(vocab, setOf(B), listOf(Trial(mapOf(A to 0, B to 0, C to 0), 20.0))),
                    RLeaf(vocab, setOf(B), listOf(Trial(mapOf(A to 1, B to 0, C to 0), 25.0))),
                    stale = false
                )
            )

        val transposedTree = transposeTree(dt, Pair(A, 0))

        val expectedPassStats = createStats(vocab, listOf(
            Trial(mapOf(A to 0, B to 0, C to 1), 10.0),
            Trial(mapOf(A to 0, B to 0, C to 0), 20.0)
        ))

        val expectedFailStats = createStats(vocab, listOf(
            Trial(mapOf(A to 1, B to 0, C to 1), 15.0),
            Trial(mapOf(A to 1, B to 0, C to 0), 25.0)
        ))

        val expectedTree =
            RDecision(vocab, vocab,
                topLevelStats,
                Pair(A, 0),
                RDecision(vocab, vocab - A,
                    expectedPassStats,
                    Pair(C, 1),
                    RLeaf(vocab, setOf(B), listOf(Trial(mapOf(A to 0, B to 0, C to 1), 10.0))),
                    RLeaf(vocab, setOf(B), listOf(Trial(mapOf(A to 0, B to 0, C to 0), 20.0))),
                    stale = true
                ),
                RDecision(vocab, vocab - A,
                    expectedFailStats,
                    Pair(C, 1),
                    RLeaf(vocab, setOf(B), listOf(Trial(mapOf(A to 1, B to 0, C to 1), 15.0))),
                    RLeaf(vocab, setOf(B), listOf(Trial(mapOf(A to 1, B to 0, C to 0), 25.0))),
                    stale = true
                )
            )
        Assert.assertEquals(expectedTree, transposedTree)
    }

    @Test
    fun transpose_recursive_correctTree(){
        val vocab = setOf(A, B, C)
        val topLevelStats = createStats(vocab, listOf(
            Trial(mapOf(A to 0, B to 0, C to 0), 10.0),
            Trial(mapOf(A to 1, B to 1, C to 1), 15.0),
            Trial(mapOf(A to 0, B to 1, C to 0), 20.0)
        ))

        val childStats = createStats(vocab, listOf(
            Trial(mapOf(A to 1, B to 1, C to 1), 15.0),
            Trial(mapOf(A to 0, B to 1, C to 0), 20.0)
        ))

        val dt = RDecision(
            vocab,
            vocab,
            topLevelStats,
            Pair(B, 0),
            RLeaf(vocab, vocab - B, listOf(Trial(mapOf(A to 0, B to 0, C to 0), 10.0))),
            RDecision(
                vocab,
                vocab - B,
                childStats,
                Pair(C, 1),
                RLeaf(vocab, setOf(A), listOf(Trial(mapOf(A to 1, B to 1, C to 1), 15.0))),
                RLeaf(vocab, setOf(A), listOf(Trial(mapOf(A to 0, B to 1, C to 0), 20.0)))
            )
        )

        val transposedTree = transposeTree(dt, RVTest(A, 0))

        val expectedChildStats = createStats(vocab, listOf(
            Trial(mapOf(A to 0, B to 0, C to 0), 10.0),
            Trial(mapOf(A to 0, B to 1, C to 0), 20.0)
        ))

        val expected = RDecision(
            vocab,
            vocab,
            topLevelStats,
            Pair(A, 0),
            RDecision(
                vocab,
                vocab - A,
                expectedChildStats,
                Pair(B, 0),
                RLeaf(vocab, setOf(C), listOf(Trial(mapOf(A to 0, B to 0, C to 0), 10.0))),
                RLeaf(vocab, setOf(C), listOf(Trial(mapOf(A to 0, B to 1, C to 0), 20.0)))
            ),
            RLeaf(
                vocab,
                vocab - A,
                listOf(
                    Trial(mapOf(A to 1, B to 1, C to 1), 15.0)
                )
            ),
            stale = true
        )

        Assert.assertEquals(expected, transposedTree)
    }
}