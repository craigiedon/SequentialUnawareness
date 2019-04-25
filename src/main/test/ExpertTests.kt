import org.junit.Assert
import org.junit.Test

class ExpertTests{

    private val rewardTree = DTDecision(Pair(A,0),
        DTDecision(Pair(B,0),
            DTLeaf(5.0),
            DTLeaf(10.0)
        ),
        DTLeaf(3.0)
    )

    private val trueMDP = MDP(setOf(A,B,C,D), rewardTree, setOf("A1", "A2"), emptyMap(), 0.99)

    @Test
    fun rewardScope_noKnownScopeNoAgentVocab_singleRandomVocab(){
        val knownScope = setOf<RandomVariable>()
        val knownAgentVocab = HashSet<RandomVariable>()

        val expert = Expert(10, 0.1, 1, 100, knownAgentVocab, trueMDP, emptyMap())
        val result = whatsInRewardScope(knownScope, expert)

        Assert.assertTrue(result == setOf(A) || result == setOf(B))
    }

    @Test(expected = IllegalArgumentException::class)
    fun rewardScope_knownScopeIsTrueScope_Exception(){
        val knownScope = setOf(A,B)
        val expert = Expert(10, 0.1, 1, 100, HashSet(), trueMDP, emptyMap())
        whatsInRewardScope(knownScope, expert)
    }

    @Test
    fun rewardScope_knownVocabRewardScopeKnown_addOneUnknownVocab(){
        val knownScope = setOf(A)
        val knownAgentVocab = mutableSetOf(A)

        val expert = Expert(10, 0.1, 1, 100, knownAgentVocab, trueMDP, emptyMap())
        val result = whatsInRewardScope(knownScope, expert)
        Assert.assertEquals(setOf(B), result)
    }

    @Test
    fun rewardScope_allVocabKnownRewardScopeUnknown_addAllRemainingScope(){
        val knownScope = emptySet<RandomVariable>()
        val knownAgentVocab = mutableSetOf(A, B)

        val expert = Expert(10, 0.1, 1, 100, knownAgentVocab, trueMDP, emptyMap())
        val result = whatsInRewardScope(knownScope, expert)
        Assert.assertEquals(setOf(A, B), result)
    }

    @Test
    fun betterActionAdvice_leafQs_adviceGivenIsBestAction(){
        val qFuncs = mapOf("A1" to DTLeaf(3.0), "A2" to DTLeaf(5.0), "A3" to DTLeaf(4.0))
        val expert = Expert(10, 0.1, 1, 100, mutableSetOf(), trueMDP, qFuncs)
        val result = betterActionAdvice(mapOf(A to 0, B to 0), expert)
        Assert.assertEquals("A2", result)
    }

    @Test
    fun betterActionAdvice_decisionQs_adviceGivenIsBestAction(){
        val qFuncs = mapOf("A1" to DTDecision(Pair(A,0), DTLeaf(3.0), DTLeaf(100.0)), "A2" to DTDecision(Pair(A,0), DTLeaf(5.0), DTLeaf(0.0)), "A3" to DTLeaf(4.0))
        val expert = Expert(10, 0.1, 1, 100, mutableSetOf(), trueMDP, qFuncs)
        val result = betterActionAdvice(mapOf(A to 1, B to 0), expert)
        Assert.assertEquals("A1", result)
    }

    @Test
    fun poorPerformance_noHistory_false(){
        val result = poorRecentPerformance(emptyList(), 10, 0.2)
        Assert.assertFalse(result)
    }

    @Test
    fun poorPerformance_shortenedActionHistory_judgeOnWhatsThere(){
        val result = poorRecentPerformance(listOf(false, false), 10, 0.2)
        Assert.assertTrue(result)
    }

    @Test
    fun poorPerformance_overMistakeThreshold_true(){
        val result = poorRecentPerformance(listOf(false, false, false, true), 4, 0.2)
        Assert.assertTrue(result)
    }

    @Test
    fun poorPerformance_underMistakeThreshold_false(){
        val result = poorRecentPerformance(listOf(false, false, false, true), 4, 0.9)
        Assert.assertFalse(result)
    }

    @Test
    fun poorPerformance_onMistakeThreshold_true(){
        val result = poorRecentPerformance(listOf(false, false, false, true), 4, 0.25)
        Assert.assertTrue(result)
    }

    @Test(expected = IllegalArgumentException::class)
    fun resolveMisunderstanding_agentShouldKnowDifference_exception(){
        val expert = Expert(10, 0.1, 1, 100, mutableSetOf(A), trueMDP, emptyMap())
        expert.episodeHistory.add(mutableListOf(
            SequentialTrial(mapOf(A to 0, B to 0), "A1", emptyMap(), 5.0),
            SequentialTrial(mapOf(A to 1, B to 0),  "A1",emptyMap(), 5.0)
        ))
        resolveMisunderstanding(0, 1, expert)
    }

    @Test(expected = IllegalArgumentException::class)
    fun resolveMisunderstanding_trialsIdentical_exception(){
        val expert = Expert(10, 0.1, 1, 100, mutableSetOf(), trueMDP, emptyMap())
        expert.episodeHistory.add(mutableListOf(
            SequentialTrial(mapOf(A to 0, B to 0), "A1",emptyMap(),  5.0),
            SequentialTrial(mapOf(A to 0, B to 0), "A1",emptyMap(),  5.0)
        ))
        resolveMisunderstanding(0, 1, expert)
    }

    @Test
    fun resolveMisunderstanding_singleVocabDifference_returnDifference(){
        val expert = Expert(10, 0.1, 1, 100, mutableSetOf(A), trueMDP, emptyMap())
        expert.episodeHistory.add(mutableListOf(
            SequentialTrial(mapOf(A to 0, B to 0), "A1",emptyMap(),  5.0),
            SequentialTrial(mapOf(A to 0, B to 1), "A1",emptyMap(),  5.0)
        ))
        val result = resolveMisunderstanding(0, 1, expert)
        Assert.assertEquals(MisunderstandingResolution(Pair(B, 0), Pair(B, 1)), result)
    }

    @Test
    fun resolveMisunderstanding_twoVocabDifference_returnEither(){
        val expert = Expert(10, 0.1, 1, 100, mutableSetOf(), trueMDP, emptyMap())
        expert.episodeHistory.add(mutableListOf(
            SequentialTrial(mapOf(A to 0, B to 0), "A1",emptyMap(),  5.0),
            SequentialTrial(mapOf(A to 1, B to 1), "A1",emptyMap(),  5.0)
        ))
        val result = resolveMisunderstanding(0, 1, expert)
        val possibleAnswer1 = MisunderstandingResolution(Pair(A,0), Pair(A, 1))
        val possibleAnswer2 = MisunderstandingResolution(Pair(B,0), Pair(B, 1))

        Assert.assertTrue(result == possibleAnswer1 || result == possibleAnswer2)
    }

    @Test(expected = IllegalArgumentException::class)
    fun globalToLocal_EmptyHistory_throwError(){
        val (epNum, localTime) = globalToLocal(listOf(emptyList<Trial>()), 3)
    }

    @Test(expected = IllegalArgumentException::class)
    fun globalToLocal_TimeGreaterThanHistory_throwError(){
        val (epNum, localTime) = globalToLocal(listOf(listOf(1,2)), 3)
    }

    @Test
    fun globalToLocal_ExactlyOnEpisodeStart_EpNumAndZero(){
        val (epNum, localTime) = globalToLocal(listOf(
            listOf(1,2,3),
            listOf(4,5,6)
        ), 3)
        Assert.assertEquals(1, epNum)
        Assert.assertEquals(0, localTime)
    }

    @Test
    fun globalToLocal_PartWayThroughEpisode_EpNumAndLocal(){
        val (epNum, localTime) = globalToLocal(listOf(
            listOf(1,2,3),
            listOf(4,5,6)
        ), 5)
        Assert.assertEquals(1, epNum)
        Assert.assertEquals(2, localTime)
    }

    @Test
    fun policyErrorEstimate_emptyHistory_zeroError(){
        val startStates = allAssignments(trueMDP.vocab.toList())
            .filter { trueMDP.startStateDescriptions.isEmpty() || trueMDP.startStateDescriptions.any { startDesc -> partialMatch(startDesc, it) } }
        policyErrorEstimate(emptyList(), 0.9, trueMDP, startStates, DTLeaf(5.0))
    }

    @Test
    fun policyErrorEstimate_lastEpisodeEmpty_zeroErrorForThatEpisode(){
        val startStates = allAssignments(trueMDP.vocab.toList())
            .filter { trueMDP.startStateDescriptions.isEmpty() || trueMDP.startStateDescriptions.any { startDesc -> partialMatch(startDesc, it) } }
        policyErrorEstimate(listOf(emptyList()), 0.9, trueMDP, startStates, DTLeaf(5.0))
    }
}
