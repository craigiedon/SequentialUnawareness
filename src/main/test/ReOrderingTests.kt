import org.junit.Assert
import org.junit.Test

class ReOrderingTests{
    @Test
    fun getAllTests_leaf_emptyList(){
        val result = getAllTests(DTLeaf(1.0))
        Assert.assertEquals(emptySet<RVTest>(), result)
    }

    @Test
    fun getAllTests_singleDT_singleVocab(){
        val result = getAllTests(DTDecision(Pair(A,0), DTLeaf(1.0), DTLeaf(2.0)))
        Assert.assertEquals(setOf(Pair(A,0)), result)
    }

    @Test
    fun getAllTests_multiDT_manyVocabNoDuplicates(){
        val result = getAllTests(DTDecision(Pair(A,0),
            DTDecision(Pair(B,0), DTLeaf(1.0), DTLeaf(1.0)),
            DTDecision(Pair(B,0), DTLeaf(2.0), DTLeaf(2.0))
        ))
        Assert.assertEquals(setOf(Pair(A,0),Pair(B,0)), result)
    }

    @Test(expected = IllegalArgumentException::class)
    fun spanEntropy_emptyList_throwException(){
        spanEntropy(emptyList())
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
        val dt = DTDecision(Pair(A, 0),
            DTDecision(Pair(B, 0), DTLeaf(Range(1.0,1.0)), DTLeaf(Range(1.0, 1.0))),
            DTDecision(Pair(B, 0), DTLeaf(Range(2.0, 2.0)), DTLeaf(Range(2.0, 2.0)))
        )

        val result = transposeTree(dt, Pair(A,0))
        Assert.assertEquals(dt, result)
    }

    @Test
    fun transposeTree_decisionTwoLeafs_returnNewDTsAtLeaves(){
        val dt = DTDecision(Pair(A, 0),
            DTLeaf(Range(1.0, 3.0)),
            DTLeaf(Range(2.0, 5.0))
        )

        val expected = DTDecision(Pair(B, 0),
            DTDecision(Pair(A,0),
                DTLeaf(Range(1.0, 3.0)),
                DTLeaf(Range(2.0, 5.0))
            ),
            DTDecision(Pair(A,0),
                DTLeaf(Range(1.0, 3.0)),
                DTLeaf(Range(2.0, 5.0))
            )
        )

        val result = transposeTree(dt, Pair(B,0))
        Assert.assertEquals(expected, result)
    }

    @Test
    fun transposeTree_decisionLeftLeaf_returnSwappedDT(){
        val dt = DTDecision(Pair(A, 0),
            DTDecision(Pair(B, 0), DTLeaf(Range(1.0, 2.0)), DTLeaf(Range(3.0, 4.0))),
            DTLeaf(Range(5.0, 10.0))
        )

        val expected = DTDecision(Pair(B, 0),
            DTDecision(Pair(A, 0), DTLeaf(Range(1.0, 2.0)), DTLeaf(Range(5.0,10.0))),
            DTDecision(Pair(A, 0), DTLeaf(Range(3.0, 4.0)), DTLeaf(Range(5.0,10.0)))
        )
        val result = transposeTree(dt, Pair(B,0))
        Assert.assertEquals(expected, result)
    }

    @Test
    fun transposeTree_decisionRightLeaf_sameAsAbove(){
        val dt = DTDecision(Pair(A, 0),
            DTLeaf(Range(5.0, 10.0)),
            DTDecision(Pair(B, 0), DTLeaf(Range(1.0, 2.0)), DTLeaf(Range(3.0, 4.0)))
        )

        val expected = DTDecision(Pair(B, 0),
            DTDecision(Pair(A, 0), DTLeaf(Range(5.0,10.0)), DTLeaf(Range(1.0, 2.0))),
            DTDecision(Pair(A, 0), DTLeaf(Range(5.0,10.0)), DTLeaf(Range(3.0,4.0)))
        )


        val result = transposeTree(dt, Pair(B,0))
        Assert.assertEquals(expected, result)
    }

    @Test
    fun transposeTree_decisionTreeWithRepleacementsBelow_switcheroo(){
        val dt = DTDecision(Pair(A, 0),
            DTDecision(Pair(B, 0),
                DTLeaf(Range(1.0, 2.0)),
                DTLeaf(Range(3.0, 4.0))
            ),
            DTDecision(Pair(B, 0),
                DTLeaf(Range(5.0, 6.0)),
                DTLeaf(Range(7.0, 8.0))
            )
        )

        val expected = DTDecision(Pair(B, 0),
            DTDecision(Pair(A, 0),
                DTLeaf(Range(1.0,2.0)),
                DTLeaf(Range(5.0,6.0))
            ),
            DTDecision(Pair(A, 0),
                DTLeaf(Range(3.0, 4.0)),
                DTLeaf(Range(7.0, 8.0))
            )
        )

        val result = transposeTree(dt, Pair(B,0))
        Assert.assertEquals(expected, result)
    }

    @Test
    fun rangesForVarSplits_unevenDT_returnsAppropriateStats(){
        val dt = DTDecision(Pair(A, 0),
            DTDecision(Pair(B, 0),
                DTLeaf(Range(3.0, 4.0)),
                DTLeaf(Range(5.0, 6.0))
            ),
            DTLeaf(Range(1.0, 2.0))
        )

        val expected = mapOf(
            Pair(A,0) to listOf(Range(1.0, 2.0), Range(3.0, 6.0)),
            Pair(B,0) to listOf(Range(1.0, 6.0), Range(1.0, 4.0))
        )
        val result = rangesForVarSplits(setOf(Pair(A,0), Pair(B,0)), dt)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun reOrderDT_leaf_returnLeaf(){
        val dt = DTLeaf(Range(1.0, 2.0))
        val result = reOrderDT(dt)
        Assert.assertEquals(dt, result)
    }

    @Test
    fun reOrderDT_DTJustLeafs_returnIdentical(){
        val dt = DTDecision(Pair(A, 0), DTLeaf(Range(1.0, 2.0)), DTLeaf(Range(3.0, 4.0)))
        val result = reOrderDT(dt)
        Assert.assertEquals(dt, result)
    }

    @Test
    fun reOrderDT_alreadyBest_returnIdentical(){
        val dt = DTDecision(Pair(A, 0),
            DTDecision(Pair(B, 0),
                DTLeaf(Range(1.0, 2.0)),
                DTLeaf(Range(3.0, 4.0))
            ),
            DTDecision(Pair(B, 0),
                DTLeaf(Range(10.0, 12.0)),
                DTLeaf(Range(13.0, 14.0))
            )
        )
        val result = reOrderDT(dt)
        Assert.assertEquals(dt, result)
    }

    @Test
    fun reOrderDT_topChanges_tranpose(){
        val dt = DTDecision(Pair(A, 0),
            DTDecision(Pair(B, 0),
                DTLeaf(Range(1.0, 2.0)),
                DTLeaf(Range(10.0, 11.0))
            ),
            DTDecision(Pair(B, 0),
                DTLeaf(Range(1.0, 2.0)),
                DTLeaf(Range(10.0, 11.0))
            )
        )

        val expected = DTDecision(Pair(B, 0),
            DTLeaf(Range(1.0, 2.0)),
            DTLeaf(Range(10.0, 11.0))
        )

        val result = reOrderDT(dt)
        Assert.assertEquals(expected, result)
    }


    @Test
    fun reOrderDT_smallValues_resultAsInPreviousTest(){
        val dt = DTDecision(Pair(A, 0),
            DTDecision(Pair(B, 0),
                DTLeaf(Range(0.01, 0.02)),
                DTLeaf(Range(0.1, 0.11))
            ),
            DTDecision(Pair(B, 0),
                DTLeaf(Range(0.01, 0.02)),
                DTLeaf(Range(0.1, 0.11))
            )
        )

        val expected = DTDecision(Pair(B, 0),
            DTLeaf(Range(0.01, 0.02)),
            DTLeaf(Range(0.1, 0.11))
        )

        val result = reOrderDT(dt)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun reOrderDT_smallValuesNonZeroDecimalRange_resultAsInPreviousTest(){
        val dt = DTDecision(Pair(A, 0),
            DTDecision(Pair(B, 0),
                DTLeaf(Range(0.01, 0.02)),
                DTLeaf(Range(0.1, 0.11))
            ),
            DTDecision(Pair(B, 0),
                DTLeaf(Range(0.02, 0.04)),
                DTLeaf(Range(0.2, 0.22))
            )
        )

        val expected = DTDecision(Pair(B, 0),
            DTDecision(Pair(A, 0),
                DTLeaf(Range(0.01, 0.02)),
                DTLeaf(Range(0.02, 0.04))
            ),
            DTDecision(Pair(A, 0),
                DTLeaf(Range(0.1, 0.11)),
                DTLeaf(Range(0.2, 0.22))
            )
        )

        val result = reOrderDT(dt)
        Assert.assertEquals(expected, result)
    }

    @Test
    fun reOrderDT_smallValuesAtomicLeaves_resultAsInPreviousTest(){
        val dt = DTDecision(Pair(A, 0),
            DTDecision(Pair(B, 0),
                DTLeaf(Range(1.0, 1.0)),
                DTLeaf(Range(10.0, 10.0))
            ),
            DTDecision(Pair(B, 0),
                DTLeaf(Range(3.0, 3.0)),
                DTLeaf(Range(16.0, 16.0))
            )
        )

        val expected = DTDecision(Pair(B, 0),
            DTDecision(Pair(A, 0),
                DTLeaf(Range(1.0, 1.0)),
                DTLeaf(Range(3.0, 3.0))
            ),
            DTDecision(Pair(A, 0),
                DTLeaf(Range(10.0, 10.0)),
                DTLeaf(Range(16.0, 16.0))
            )
        )

        val result = reOrderDT(dt)
        Assert.assertEquals(expected, result)
    }
}