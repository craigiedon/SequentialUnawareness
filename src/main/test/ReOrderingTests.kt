import org.junit.Assert
import org.junit.Test

class ReOrderingTests{
    @Test
    fun getVocab_leaf_emptyList(){
        val result = getVocab(DTLeaf(1.0))
        Assert.assertEquals(emptySet<RandomVariable>(), result)
    }

    @Test
    fun getVocab_singleDT_singleVocab(){
        val result = getVocab(DTDecision(A, listOf(DTLeaf(1.0), DTLeaf(2.0))))
        Assert.assertEquals(setOf(A), result)
    }

    @Test
    fun getVocab_multiDT_manyVocabNoDuplicates(){
        val result = getVocab(DTDecision(A, listOf(
            DTDecision(B, listOf(DTLeaf(1.0), DTLeaf(1.0))),
            DTDecision(B, listOf(DTLeaf(2.0), DTLeaf(2.0)))
        )))
        Assert.assertEquals(setOf(A,B), result)
    }

    @Test(expected = IllegalArgumentException::class)
    fun spanEntropy_emptyList_throwException(){
        val result = spanEntropy(emptyList())
    }

    @Test
    fun spanEntropy_zeroSpan_infiniteEntropy(){
        val result = spanEntropy(listOf(0.0))
        Assert.assertTrue(result.isInfinite() && result > 0)
    }

    @Test
    fun spanEntropy_wholeNumSpan_negativeEntropy(){
        val result = spanEntropy(listOf(10.0))
        Assert.assertTrue(result < 0)
    }

    @Test
    fun spanEntropy_decimalSpan_positiveEntropy(){
        val result = spanEntropy(listOf(0.2))
        Assert.assertTrue(result > 0)
    }


    @Test
    fun transposeTree_sameTest_returnSameDT(){
        val dt = DTDecision(A, listOf(
            DTDecision(B, listOf(DTLeaf(Pair(1.0,1.0)), DTLeaf(Pair(1.0, 1.0)))),
            DTDecision(B, listOf(DTLeaf(Pair(2.0, 2.0)), DTLeaf(Pair(2.0, 2.0))))
        ))

        val result = transposeTree(dt, A)
        Assert.assertEquals(dt, result)
    }

    @Test
    fun transposeTree_decisionTwoLeafs_returnLeaf(){
        val dt = DTDecision(A, listOf(
            DTLeaf(Pair(1.0, 3.0)),
            DTLeaf(Pair(2.0, 5.0))
        ))
        val result = transposeTree(dt, B)
        Assert.assertEquals(DTLeaf(Pair(1.0, 5.0)), result)
    }

    @Test
    fun transposeTree_decisionLeftLeaf_returnSwappedDT(){
        val dt = DTDecision(A, listOf(
            DTDecision(B, listOf(DTLeaf(Pair(1.0, 2.0)), DTLeaf(Pair(3.0, 4.0)))),
            DTLeaf(Pair(5.0, 10.0))
        ))

        val expected = DTDecision(B, listOf(
            DTDecision(A, listOf(DTLeaf(Pair(1.0, 2.0)), DTLeaf(Pair(5.0,10.0)))),
            DTDecision(A, listOf(DTLeaf(Pair(3.0, 4.0)), DTLeaf(Pair(5.0,10.0))))
        ))
        val result = transposeTree(dt, B)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun transposeTree_decisionRightLeaf_sameAsAbove(){
        val dt = DTDecision(A, listOf(
            DTLeaf(Pair(5.0, 10.0)),
            DTDecision(B, listOf(DTLeaf(Pair(1.0, 2.0)), DTLeaf(Pair(3.0, 4.0))))
        ))

        val expected = DTDecision(B, listOf(
            DTDecision(A, listOf(DTLeaf(Pair(5.0,10.0)), DTLeaf(Pair(1.0, 2.0)))),
            DTDecision(A, listOf(DTLeaf(Pair(5.0,10.0)), DTLeaf(Pair(3.0,4.0)))))
        )

        val result = transposeTree(dt, B)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun transposeTree_decisionTreeWithRepleacementsBelow_switcheroo(){
        val dt = DTDecision(A, listOf(
            DTDecision(B, listOf(
                DTLeaf(Pair(1.0, 2.0)),
                DTLeaf(Pair(3.0, 4.0)))
            ),
            DTDecision(B, listOf(
                DTLeaf(Pair(5.0, 6.0)),
                DTLeaf(Pair(7.0, 8.0))
            ))
        ))

        val expected = DTDecision(B, listOf(
            DTDecision(A, listOf(
                DTLeaf(Pair(1.0,2.0)),
                DTLeaf(Pair(5.0,6.0))
            )),
            DTDecision(A, listOf(
                DTLeaf(Pair(3.0, 4.0)),
                DTLeaf(Pair(7.0, 8.0))
            ))
        ))

        val result = transposeTree(dt, B)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun rangesForVarSplits_unevenDT_returnsAppropriateStats(){
        val dt = DTDecision(A, listOf(
            DTDecision(B, listOf(
                DTLeaf(Pair(3.0, 4.0)),
                DTLeaf(Pair(5.0, 6.0))
            )),
            DTLeaf(Pair(1.0, 2.0))
        ))

        val expected = mapOf(
            A to listOf(Pair(3.0, 6.0), Pair(1.0, 2.0)),
            B to listOf(Pair(1.0, 4.0), Pair(1.0, 6.0))
        )
        val result = rangesForVarSplits(setOf(A, B), dt)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun reOrderDT_leaf_returnLeaf(){
        val dt = DTLeaf(Pair(1.0, 2.0))
        val result = reOrderDT(dt)
        Assert.assertEquals(dt, result)
    }

    @Test
    fun reOrderDT_DTJustLeafs_returnIdentical(){
        val dt = DTDecision(A, listOf(DTLeaf(Pair(1.0, 2.0)), DTLeaf(Pair(3.0, 4.0))))
        val result = reOrderDT(dt)
        Assert.assertEquals(dt, result)
    }

    @Test
    fun reOrderDT_alreadyBest_returnIdentical(){
        val dt = DTDecision(A, listOf(
            DTDecision(B, listOf(
                DTLeaf(Pair(1.0, 2.0)),
                DTLeaf(Pair(3.0, 4.0))
            )),
            DTDecision(B, listOf(
                DTLeaf(Pair(10.0, 12.0)),
                DTLeaf(Pair(13.0, 14.0))
            ))

        ))
        val result = reOrderDT(dt)
        Assert.assertEquals(dt, result)
    }

    @Test
    fun reOrderDT_topChanges_tranpose(){
        val dt = DTDecision(A, listOf(
            DTDecision(B, listOf(
                DTLeaf(Pair(1.0, 2.0)),
                DTLeaf(Pair(10.0, 11.0))
            )),
            DTDecision(B, listOf(
                DTLeaf(Pair(1.0, 2.0)),
                DTLeaf(Pair(10.0, 11.0))
            ))

        ))

        val expected = DTDecision(B, listOf(
            DTDecision(A, listOf(
                DTLeaf(Pair(1.0, 2.0)),
                DTLeaf(Pair(1.0, 2.0))
            )),
            DTDecision(A, listOf(
                DTLeaf(Pair(10.0, 11.0)),
                DTLeaf(Pair(10.0, 11.0))
            ))

        ))

        val result = reOrderDT(dt)
        Assert.assertEquals(expected, result)
    }
}