import Utils.repeatList
import Utils.repeatNum
import org.junit.Assert
import org.junit.Test

class BuntineTests{
    @Test
    fun bestParent_singleParent_chooseThisParent(){
        val singleParent = SeqPInfo(A, setOf(B), -10.0, emptyMap(), emptyMap())
        val parentChoices = mapOf(A to listOf(singleParent))
        val result = bestParents(parentChoices)
        Assert.assertEquals(mapOf(A to singleParent), result)
    }

    @Test
    fun bestParent_multipleParents_pickHighestProb(){
        val pChoice1 = SeqPInfo(A, setOf(B), -10.0, emptyMap(), emptyMap())
        val pChoice2 = SeqPInfo(A, setOf(B, C), -3.0, emptyMap(), emptyMap())
        val parentChoices = mapOf(A to listOf(pChoice1, pChoice2))
        val result = bestParents(parentChoices)
        Assert.assertEquals(mapOf(A to pChoice2), result)
    }

    @Test(expected = NullPointerException::class)
    fun bestParent_noParents_throwException(){
        val parentChoices = mapOf(A to emptyList<SeqPInfo>())
        bestParents(parentChoices)
    }

    @Test(expected = IllegalStateException::class)
    fun revisedAliveList_emptyList_illegalStateException(){
        val originalList = listOf<PSetNode>()
        val pSetInfos = emptyMap<PSet, SeqPInfo>()
        val expected = originalList

        val result = revisedAliveList(0.1, originalList, pSetInfos)
    }

    @Test
    fun revisedAliveList_someBelowThresh_removeThoseUnderThresh(){
        val originalList = listOf(PSetNode(setOf(B)), PSetNode(setOf(C)), PSetNode(setOf(A)))
        val pSetInfos = mapOf(
            setOf(B) to SeqPInfo(A, setOf(B), Math.log(0.01), emptyMap(), emptyMap()),
            setOf(C) to SeqPInfo(A, setOf(C), Math.log(0.09), emptyMap(), emptyMap()),
            setOf(A) to SeqPInfo(A, setOf(A), Math.log(0.9), emptyMap(), emptyMap()))

        val expected = listOf(PSetNode(setOf(C)), PSetNode(setOf(A)))

        val result = revisedAliveList(0.05, originalList, pSetInfos)
        Assert.assertEquals(expected, result)
    }


    @Test
    fun structuralUpdate_emptyVocab_onlyReasonableIsEmptyList(){
        val priorJointParams = DTLeaf(Factor(listOf(A), listOf(0.1, 0.9)))
        val logPrior : (PSet) -> Double = { pSet -> -2.0 }
        val trials = emptyList<SequentialTrial>()
        val vocab = setOf<RandomVariable>()

        val expected = listOf(SeqPInfo(A, emptySet(), 0.0, emptyCounts(A, emptySet()), priorParamTable(emptySet(), priorJointParams)))

        val result = structuralUpdate(A, logPrior, vocab, trials, emptyList(), priorJointParams, 0.01, 5.0)

        Assert.assertEquals(expected, result)
    }

    @Test
    fun structuralUpdate_singleVocabReasonable_singleAndEmptyReasonable(){
        val priorJointParams = DTDecision(A, listOf(
            DTLeaf(Factor(listOf(A), listOf(0.05, 0.45))),
            DTLeaf(Factor(listOf(A), listOf(0.45, 0.05)))
        ))

        val pseudoCountStrength = 5.0
        val logPrior : (PSet) -> Double = { pSet -> Math.log(0.5)}
        val trials = emptyList<SequentialTrial>()
        val vocab = setOf(A)

        val expected = listOf(
            SeqPInfo(A, emptySet(), Math.log(0.5), emptyCounts(A, emptySet()), priorParamTable(emptySet(), priorJointParams)),
            SeqPInfo(A, setOf(A), Math.log(0.5), emptyCounts(A, setOf(A)), priorParamTable(setOf(A), priorJointParams))
        )

        val result = structuralUpdate(A, logPrior, vocab, trials, emptyList(), priorJointParams, 0.01, pseudoCountStrength)

        Assert.assertEquals(expected, result)
    }

    @Test
    fun structuralUpdate_priorDisallowsSingleVocab_onlyEmptyReasonable(){
        val priorJointParams = DTDecision(A, listOf(
            DTLeaf(Factor(listOf(A), listOf(0.05, 0.45))),
            DTLeaf(Factor(listOf(A), listOf(0.45, 0.05)))
        ))

        val pseudoCountStrength = 5.0
        val logPrior : (PSet) -> Double = { pSet ->
            when(pSet){
                emptySet<RandomVariable>() -> Math.log(1.0)
                else -> -10000000000.0
            }}
        val trials = emptyList<SequentialTrial>()
        val vocab = setOf(A)

        val expected = listOf(
            SeqPInfo(A, emptySet(), Math.log(1.0), emptyCounts(A, emptySet()), priorParamTable(emptySet(), priorJointParams))
        )

        val result = structuralUpdate(A, logPrior, vocab, trials, emptyList(), priorJointParams, 0.01, pseudoCountStrength)

        Assert.assertEquals(expected, result)
    }

    @Test
    fun structuralUpdate_twoParentsReasonableButEachSingleParentUnreasonable_emptyPSetOnly(){
        val priorJointParams = DTLeaf(Factor(listOf(A), listOf(0.1, 0.9)))
        val pseudoCountStrength = 5.0
        val logPrior : (PSet) -> Double = { pSet ->
            when(pSet){
                emptySet<RandomVariable>() -> -1000.0
                setOf(A, B) -> 1.0
                else -> -1000000000000000000000.0
            }
        }

        val trials = emptyList<SequentialTrial>()
        val vocab = setOf(A,B)

        val expected = listOf(
            SeqPInfo(A, emptySet(), Math.log(1.0), emptyCounts(A, emptySet()), priorParamTable(emptySet(), priorJointParams))
        )

        val result = structuralUpdate(A, logPrior, vocab, trials, emptyList(), priorJointParams, 0.01, pseudoCountStrength)

        Assert.assertEquals(expected, result)
    }

    @Test
    fun structuralUpdate_twoParentsVastlyPopular_everythingElseUnreasonable(){
        val priorJointParams = DTDecision(A, listOf(
            DTLeaf(Factor(listOf(A), listOf(0.1, 0.9))),
            DTDecision(B, listOf(
                DTLeaf(Factor(listOf(A), listOf(0.1,0.9))),
                DTLeaf(Factor(listOf(A), listOf(0.1,0.9)))
            ))
        ))

        val pseudoCountStrength = 5.0
        val logPrior : (PSet) -> Double = { pSet ->
            when(pSet){
                setOf(A,B) -> 0.0
                else -> -10000.0
            }
        }

        val trials = emptyList<SequentialTrial>()
        val vocab = setOf(A,B)

        val expected = listOf(
            SeqPInfo(A, setOf(A,B), Math.log(1.0), emptyCounts(A, setOf(A,B)), priorParamTable(setOf(A,B), priorJointParams))
        )

        val result = structuralUpdate(A, logPrior, vocab, trials, emptyList(), priorJointParams, 0.01, pseudoCountStrength)

        Assert.assertEquals(expected, result)
    }

    @Test
    fun structuralUpdate_twoVocabAllLikely_AllReasonableAndProbsNormalized(){
        val priorJointParams = DTLeaf(Factor(listOf(A), listOf(0.1, 0.9)))

        val pseudoCountStrength = 5.0
        val logPrior : (PSet) -> Double = { 0.0 }

        val trials = emptyList<SequentialTrial>()
        val vocab = setOf(A,B)

        val expected = listOf(
            SeqPInfo(A, setOf(), Math.log(0.25), emptyCounts(A, setOf()), priorParamTable(setOf(), priorJointParams)),
            SeqPInfo(A, setOf(A),  Math.log(0.25), emptyCounts(A, setOf(A)), priorParamTable(setOf(A), priorJointParams)),
            SeqPInfo(A, setOf(B),  Math.log(0.25), emptyCounts(A, setOf(B)), priorParamTable(setOf(B), priorJointParams)),
            SeqPInfo(A, setOf(A,B),  Math.log(0.25), emptyCounts(A, setOf(A,B)), priorParamTable(setOf(A,B), priorJointParams))
        )

        val result = structuralUpdate(A, logPrior, vocab, trials, emptyList(), priorJointParams, 0.01, pseudoCountStrength)

        Assert.assertEquals(expected, result)
    }

    @Test
    fun siblings_noParents_noSiblings(){
        val result = siblings(LatticeNode(A, ArrayList()))
        Assert.assertEquals(emptyList<LatticeNode<RandomVariable>>(), result)
    }

    @Test
    fun siblings_coupleOfSiblings_catchesSiblingsButNotStepSiblings(){
        val parentNode = LatticeNode(B, ArrayList())
        val otherParentNode = LatticeNode(D, ArrayList())

        val siblingNode = LatticeNode(C, mutableListOf(parentNode, otherParentNode))
        val stepSiblingNode = LatticeNode(D, mutableListOf(otherParentNode))

        val aNode = LatticeNode(A, mutableListOf(parentNode))
        val result = siblings(aNode)
        Assert.assertEquals(listOf(siblingNode), result)
    }

    @Test
    fun createPInfo_countsAndPseudoSetCorrectly(){
        val child = A
        val pSet = setOf(B)
        val trials = listOf(
            SequentialTrial(mapOf(A to 0, B to 0), "A1", mapOf(A to 1, B to 0), 0.0),
            SequentialTrial(mapOf(A to 1, B to 0), "A1", mapOf(A to 0, B to 0), 0.0),
            SequentialTrial(mapOf(A to 0, B to 0), "A1", mapOf(A to 1, B to 1), 0.0)
        )
        val logPrior : (PSet) -> Double = { -0.7 }

        val priorJointParams = DTLeaf(Factor(listOf(child), listOf(0.2, 0.8)))
        val priorSampleSize = 5.0
        val expertEv = emptyList<DirEdge>()

        val expectedPriorParams = allAssignments(pSet.toList()).associate { Pair(it, Factor(listOf(A), listOf(0.1, 0.4))) }
        val expectedCounts = mapOf(
            mapOf(B to 0) to mutableListOf(1,2),
            mapOf(B to 1) to mutableListOf(0,0)
        )

        val expectedLogProbability = -3.27451880847768744

        val result = createPInfo(child, pSet, trials, logPrior, priorJointParams, priorSampleSize, expertEv)

        Assert.assertEquals(child, result.child)
        Assert.assertEquals(pSet, result.parentSet)
        Assert.assertEquals(expectedPriorParams, result.priorParams)
        Assert.assertEquals(expectedCounts, result.counts)
        Assert.assertEquals(expectedLogProbability, result.logProbability, 10E-10)
    }

    @Test
    fun parameterUpdate_emptyReasonableSetIn_emptyReasonableSetOut(){
        val reasonableParents = emptyList<SeqPInfo>()
        val seqTrial = SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 1), 0.0)
        val expertEvidence = emptyList<DirEdge>()
        val alphaTotal = 5.0

        val result = parameterUpdate(A, reasonableParents, seqTrial, expertEvidence, alphaTotal)
        val expected = emptyList<SeqPInfo>()
        Assert.assertEquals(expected, result)
    }

    @Test
    fun parameterUpdate_singleEntry_UpdateCountsProbabilityCertain(){
        val logPrior = -0.5
        val oldLogLikelihood = -1.0
        val reasonableParent = SeqPInfo(A,
            emptySet(),
            logPrior + oldLogLikelihood,
            mapOf(emptyMap<RandomVariable, Int>() to mutableListOf(1, 1)),
            mapOf(emptyMap<RandomVariable, Int>() to Factor(listOf(A), listOf(0.5,0.5)))
        )

        val reasonableParents = listOf(reasonableParent)
        val seqTrial = SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 1), 0.0)
        val expertEvidence = emptyList<DirEdge>()
        val alphaTotal = 5.0

        val result = parameterUpdate(A, reasonableParents, seqTrial, expertEvidence, alphaTotal)
        Assert.assertEquals(1, result.size)
        Assert.assertEquals(emptySet<RandomVariable>(), result[0].parentSet)
        Assert.assertEquals(mapOf(emptyMap<RandomVariable, Int>() to mutableListOf(1,2)), result[0].counts)
        Assert.assertEquals(0.0, result[0].logProbability, 10E-10)
    }

    @Test
    fun parameterUpdate_twoPSets_UpdateCountsAndRenormalizeProbs(){
        val logPrior = -0.5

        val logLikelihood1 = -1.568615917913845245046182268070867769348389606634997
        val reasonableParent1 = SeqPInfo(A,
            emptySet(),
            logPrior + logLikelihood1,
            mapOf(emptyMap<RandomVariable, Int>() to mutableListOf(1, 1)),
            mapOf(emptyMap<RandomVariable, Int>() to Factor(listOf(A), listOf(0.5,0.5)))
        )

        val logLikelhood2 = -1.386294361119890618834464242916353136151000268720510
        val reasonableParent2 = SeqPInfo(A,
            setOf(A),
            logPrior + logLikelhood2,
            mapOf(
                mapOf(A to 0) to mutableListOf(0,1),
                mapOf(A to 1) to mutableListOf(1,0)
            ),
            allAssignments(listOf(A)).associate { Pair(it, Factor(listOf(A), listOf(0.25, 0.25))) }
        )

        val reasonableParents = listOf(reasonableParent1, reasonableParent2)
        val seqTrial = SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 1), 0.0)
        val expertEvidence = emptyList<DirEdge>()
        val alphaTotal = 5.0

        val result = parameterUpdate(A, reasonableParents, seqTrial, expertEvidence, alphaTotal)
        Assert.assertEquals(2, result.size)

        Assert.assertEquals(emptySet<RandomVariable>(), result[0].parentSet)
        Assert.assertEquals(setOf(A), result[1].parentSet)

        Assert.assertEquals(mapOf(emptyMap<RandomVariable, Int>() to mutableListOf(1,2)), result[0].counts)
        Assert.assertEquals(mapOf(mapOf(A to 0) to mutableListOf(0, 2), mapOf(A to 1) to mutableListOf(1, 0)), result[1].counts)


        val expectedUnnormed1 = -0.5 -2.261763098473790554463414389529044337423889740995252
        val expectedUnnormed2 = -0.5 -1.828127113398929850566558633972658024568604017017128
        val logTotal = Math.log(Math.exp(expectedUnnormed1) + Math.exp(expectedUnnormed2))

        val logProb1 = expectedUnnormed1 - logTotal
        val logProb2 = expectedUnnormed2 - logTotal
        Assert.assertEquals(logProb1, result[0].logProbability, 10E-10)
        Assert.assertEquals(logProb2, result[1].logProbability, 10E-10)
    }

    @Test
    fun trialUpdate_staleReasonableParentsButWaitingPeriodUnmet_sameReasonableParents(){
        val bUpd = BuntineUpdater(0.01, 5.0, 0.1)
        val seqTrial = SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 1), 0.0)
        val trialHistory = repeatList(
            listOf(
                SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 1), 0.0),
                SequentialTrial(mapOf(A to 1), "A1", mapOf(A to 0), 0.0)
            ),
            500
        )

        val reasonableParent1 = SeqPInfo(A,
            emptySet(),
            0.5,
            mapOf(emptyMap<RandomVariable, Int>() to mutableListOf(1, 1)),
            mapOf(emptyMap<RandomVariable, Int>() to Factor(listOf(A), listOf(0.5,0.5)))
        )

        val reasonableParent2 = SeqPInfo(A,
            setOf(A),
            0.5,
            mapOf(
                mapOf(A to 0) to mutableListOf(0,1),
                mapOf(A to 1) to mutableListOf(1,0)
            ),
            allAssignments(listOf(A)).associate { Pair(it, Factor(listOf(A), listOf(0.25, 0.25))) }
        )

        val reasonableParents = mapOf(A to listOf(reasonableParent1, reasonableParent2))
        val bestPInfos = mapOf(A to reasonableParents[A]!![0])

        val dbnInfo = DBNInfo(reasonableParents, bestPInfos, emptyMap(), emptyMap(), emptyMap(), emptyMap(), 0)
        val result = bUpd.trialUpdate(seqTrial, trialHistory, emptyList(), 1, dbnInfo)
        Assert.assertEquals(2, result.reasonableParents[A]!!.size)
    }

    @Test
    fun trialUpdate_staleReasonableParentsWaitingPeriodMet_differentReasonableParents(){
        val bUpd = BuntineUpdater(0.01, 5.0, 0.1)
        val seqTrial = SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 1), 0.0)
        val trialHistory = repeatList(
            listOf(
                SequentialTrial(mapOf(A to 0), "A1", mapOf(A to 1), 0.0),
                SequentialTrial(mapOf(A to 1), "A1", mapOf(A to 0), 0.0)
            ),
            500
        )

        val reasonableParent1 = SeqPInfo(A,
            emptySet(),
            0.5,
            mapOf(emptyMap<RandomVariable, Int>() to mutableListOf(1, 1)),
            mapOf(emptyMap<RandomVariable, Int>() to Factor(listOf(A), listOf(0.5,0.5)))
        )

        val reasonableParent2 = SeqPInfo(A,
            setOf(A),
            0.5,
            mapOf(
                mapOf(A to 0) to mutableListOf(0,1),
                mapOf(A to 1) to mutableListOf(1,0)
            ),
            allAssignments(listOf(A)).associate { Pair(it, Factor(listOf(A), listOf(0.25, 0.25))) }
        )

        val reasonableParents = mapOf(A to listOf(reasonableParent1, reasonableParent2))
        val bestPInfos = mapOf(A to reasonableParents[A]!![0])
        val pSetPriors : Map<RandomVariable, (PSet) -> Double> = mapOf(A to {pSet -> -0.5})
        val jointParamPriors = mapOf(A to DTLeaf(Factor(listOf(A), listOf(0.5, 0.5))))

        val dbnInfo = DBNInfo(reasonableParents, bestPInfos, emptyMap(), jointParamPriors, emptyMap(), pSetPriors, 0)
        val result = bUpd.trialUpdate(seqTrial, trialHistory, emptyList(), 5000, dbnInfo)
        Assert.assertEquals(1, result.reasonableParents[A]!!.size)
    }

    @Test
    fun initialDBNInfo_SingleVariable_InitialResult(){
        val bUpd = BuntineUpdater(0.001, 5.0, 0.1)
        val result = bUpd.initialDBNInfo(setOf(A), emptyList(), 0)
        Assert.assertEquals(2, result.reasonableParents[A]!!.size)
        Assert.assertEquals(emptySet<RandomVariable>(), result.bestPInfos[A]!!.parentSet)
        Assert.assertEquals(Math.log(0.9), result.pSetPriors[A]!!(emptySet()), 10E-10)
        Assert.assertEquals(Math.log(0.1), result.pSetPriors[A]!!(setOf(A)), 10E-10)
    }

    @Test
    fun posteriorToPrior_twoVocabWithOneReasonableEach_NewRVAddedOnAndMixed(){
        val jointPriorParamsOldVocab = mapOf(
            A to DTLeaf(Factor(listOf(A), listOf(0.4, 0.6))),
            B to DTLeaf(Factor(listOf(B), listOf(0.9, 0.1)))
        )

        val oldLogProbs = mapOf(
            A to mapOf(setOf(A) to Math.log(1.0)),
            B to mapOf(emptySet<RandomVariable>() to Math.log(1.0))
        )

        val result = posteriorToPrior(oldLogProbs, setOf(A, B), setOf(C), 0.1, 0.01)

        Assert.assertEquals(listOf(setOf(A), setOf(A, C)), result.reasonableParents[A]!!)
        Assert.assertEquals(Math.log(0.99 * 0.1), result.logPriors[A]!!(setOf(A, C)), 10E-10)
        Assert.assertEquals(Math.log(0.99 * 0.9), result.logPriors[A]!!(setOf(A)), 10E-10)


        Assert.assertEquals(listOf(emptySet(), setOf(C)), result.reasonableParents[B]!!)
        Assert.assertEquals(Math.log(0.99 * 0.1), result.logPriors[B]!!(setOf(C)), 10E-10)
        Assert.assertEquals(Math.log(0.99 * 0.9), result.logPriors[B]!!(setOf()), 10E-10)
        Assert.assertEquals(Math.log(0.01) + Math.log(1.0 / (8 - 2)), result.logPriors[B]!!(setOf(B)), 10E-10)
    }

    @Test
    fun addBeliefVariables_twoVocabOneNewRV_correctlyUpdatedResult(){
        val bUpd = BuntineUpdater(0.01, 5.0, 0.1)

        val jointPriorParamsOldVocab = mapOf(
            A to DTLeaf(Factor(listOf(A), listOf(0.4, 0.6))),
            B to DTLeaf(Factor(listOf(B), listOf(0.9, 0.1)))
        )

        val oldLogProbs = mapOf(
            A to mapOf(setOf(A) to Math.log(1.0)),
            B to mapOf(emptySet<RandomVariable>() to Math.log(1.0))
        )

        val AParent = SeqPInfo(A, setOf(A), Math.log(1.0), emptyMap(), emptyMap())
        val BParent = SeqPInfo(B, setOf(B), Math.log(1.0), emptyMap(), emptyMap())
        val reasonableParents = mapOf(
            A to listOf(AParent),
            B to listOf(BParent)
        )

        val bestPInfos = mapOf(A to AParent, B to BParent)
        val logPriors : Map<RandomVariable, (PSet) -> Double> = mapOf(A to { pSet -> Math.log(1.0) }, B to { pSet -> Math.log(1.0) })

        val updateConfig = ITIUpdateConfig<SequentialTrial, Int>({1}, { x, y -> true}, { x, y -> 1.0}, 1.0, setOf(A,B))
        val cptsITI = mapOf(
            A to Pair(ITILeaf(emptyMap(), emptyMap(), emptyList<SequentialTrial>(), HashMap<Int,Int>(), false), updateConfig),
            B to Pair(ITILeaf(emptyMap(), emptyMap(), emptyList<SequentialTrial>(), HashMap<Int,Int>(), false), updateConfig)
        )

        val dbnInfo = DBNInfo(reasonableParents, bestPInfos, cptsITI, jointPriorParamsOldVocab, jointPriorParamsOldVocab, logPriors, 0)
        val result = bUpd.addBeliefVariables(setOf(C), 0, emptyList(), emptyList(), dbnInfo)

        Assert.assertEquals(dbnInfo.dbn, result.priorJointParams)
        Assert.assertEquals(Math.log(0.1 * 0.1 * 0.9), result.pSetPriors[C]!!(setOf(A, B)), 10E-10)
        Assert.assertEquals(7, result.reasonableParents[C]!!.size)
    }
}
