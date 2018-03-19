import Utils.allEqual
import Utils.map
import org.apache.commons.math3.special.Gamma

data class SequentialTrial(val prevState : RVAssignment, val action : Action, val currentState : RVAssignment, val reward : Double)

interface ITITree<T, C>{
    fun incrementalUpdate(examples: List<T>) : ITITree<T, C>
    fun changeAllowedVocab(allowedVocab : Set<RandomVariable>) : ITITree<T, C>
    val rootNode : ITINode<T,C>
}

sealed class ITINode<T, C>{
    abstract val branchLabel : RVAssignment
    abstract val testCandidates : Map<RVTest, TestStats<C>>
    abstract val stale : Boolean
}
data class ITILeaf<T, C>(
    override val branchLabel : RVAssignment,
    override val testCandidates: Map<RVTest, TestStats<C>>,
    val examples : List<T>,
    val counts : MutableMap<C, Int>,
    override val stale : Boolean
) : ITINode<T, C>()

data class ITIDecision<T, C>(
    override val branchLabel : RVAssignment,
    override val testCandidates: Map<RVTest, TestStats<C>>,
    val currentTest: RVTest,
    val passBranch : ITINode<T,C>,
    val failBranch : ITINode<T,C>,
    override val stale : Boolean
) : ITINode<T, C>()

typealias ProbNode = ITINode<SequentialTrial, Int>
typealias PLeaf = ITILeaf<SequentialTrial, Int>
typealias PDecision = ITIDecision<SequentialTrial, Int>

data class ProbTreeITI(val rv : RandomVariable, val vocab : Set<RandomVariable>, val jointParamPrior : DecisionTree<Factor>, val pseudoCountSize : Double, var root: ProbNode? = null) : ITITree<SequentialTrial, Int>{

    fun exampleToClass(example : SequentialTrial) = example.currentState[rv]!!
    fun testChecker(rvTest: RVTest, example : SequentialTrial) = example.prevState[rvTest.first] == rvTest.second
    fun scoreFunc(partialAssignment: RVAssignment, classCounts : Map<Int, Int>) = BDeScore(rv, classCounts, partialAssignment, jointParamPrior, pseudoCountSize)
    val splitThresh = 0.7

    init{
        if(root == null){
            val emptyRoot = PLeaf(emptyMap(), createStats(vocab, emptyList(), this::testChecker, this::exampleToClass), emptyList(), HashMap(), stale = true)
            root = ensureBestTest(emptyRoot, vocab, this::scoreFunc, splitThresh, this::testChecker, this::exampleToClass)
        }
    }

    override fun changeAllowedVocab(allowedVocab: Set<RandomVariable>) : ProbTreeITI{
        fun changeAllowedVocabRec(pt : ProbNode, allowedVocab: Set<RandomVariable>) : ProbNode{
            if(vocab == allowedVocab){
                return pt
            }
            val additionalVars = allowedVocab - vocab
            val filteredTests = pt.testCandidates.filterKeys{ it.first in allowedVocab }
            when(pt){
                is PLeaf -> {
                    val additionalTests = createStats(additionalVars, pt.examples, this::testChecker, this::exampleToClass)
                    val updatedTests = filteredTests + additionalTests
                    return pt.copy(testCandidates = updatedTests)
                }
                is PDecision -> {
                    val newPassBranch = changeAllowedVocabRec(pt.passBranch, allowedVocab)
                    val newFailBranch = changeAllowedVocabRec(pt.failBranch, allowedVocab)
                    if(additionalVars.isEmpty() && pt.currentTest.first in allowedVocab){
                        return pt.copy(testCandidates = filteredTests,  passBranch = newPassBranch, failBranch = newFailBranch)
                    }
                    val extraPassTests = newPassBranch.testCandidates.filterKeys { it.first in additionalVars }
                    val extraFailTests = newFailBranch.testCandidates.filterKeys { it.first in additionalVars }
                    val additionalTests = mergeTests(extraPassTests, extraFailTests)
                    return pt.copy(testCandidates = filteredTests + additionalTests,  passBranch = newPassBranch, failBranch = newFailBranch, stale = true)
                }
            }
        }
        return this.copy(vocab = allowedVocab, root = changeAllowedVocabRec(root!!, allowedVocab))
    }

    override val rootNode get() = root!!

    override fun incrementalUpdate(examples : List<SequentialTrial>): ProbTreeITI {
        val exampleAddedTree = addExamples(root!!, examples, this::exampleToClass, this::testChecker)
        val testUpdatedNode = ensureBestTest(exampleAddedTree, this.vocab, this::scoreFunc, splitThresh, this::testChecker, this::exampleToClass)
        return this.copy(root = testUpdatedNode)
    }

}

fun <T, C> addExamples(pTree: ITINode<T,C>, examples: List<T>, classExtractor : (T) -> C, testChecker : (RVTest, T) -> Boolean) : ITINode<T, C> {
    when(pTree){
        is ITILeaf -> {
            // Update Counts
            for (example in examples) {
                val classLabel = classExtractor(example)
                val oldVal : Int = pTree.counts.getOrDefault(classLabel, 0)
                pTree.counts[classLabel] = oldVal + 1
            }
            pTree.testCandidates.forEach { _, testStat -> addTrials(testStat, examples, testChecker, classExtractor) }
            return pTree.copy(stale = true)
            //return attemptSplit(pTree, vocab, testChecker, classExtractor, scoreFunc, splitThresh)
        }
        is ITIDecision -> {
            val (passTrials, failTrials) = examples.partition { testChecker(pTree.currentTest, it) }
            val newPass = addExamples(pTree.passBranch, passTrials, classExtractor, testChecker)
            val newFail = addExamples(pTree.failBranch, failTrials, classExtractor, testChecker)

            val updatedDT = pTree.copy(passBranch = newPass, failBranch = newFail, stale = true)
            updatedDT.testCandidates.forEach { (_, stat) -> addTrials(stat, examples, testChecker, classExtractor) }
            return  updatedDT
        }
    }
}

fun <T, C> ensureBestTest(dt: ITINode<T, C>, vocab : Set<RandomVariable>, scoreFunc: (RVAssignment, Map<C, Int>) -> Double, scoreThresh : Double, testChecker : (RVTest, T) -> Boolean, classExtractor: (T) -> C) : ITINode<T,C>{
    if(dt.stale){
        val testScores = dt.testCandidates.mapValues{ (_, testStat) ->
            scoreFunc(dt.branchLabel + Pair(testStat.rv, testStat.testVal), testStat.passTrialCounts) +
            scoreFunc(dt.branchLabel + Pair(testStat.rv, 1 - testStat.testVal), testStat.failTrialCounts)
        }

        val (bestTest, bestScore) = testScores.entries.maxBy { it.value }!!

        when(dt){
            is ITIDecision -> {

                val revisedNode = if(bestTest != dt.currentTest && bestScore - testScores[dt.currentTest]!! > scoreThresh)
                    transposeTree(dt, bestTest, vocab, testChecker, classExtractor)
                else dt

                return revisedNode.copy(stale = false,
                    passBranch = ensureBestTest(revisedNode.passBranch, vocab, scoreFunc, scoreThresh, testChecker, classExtractor),
                    failBranch = ensureBestTest(revisedNode.failBranch, vocab, scoreFunc, scoreThresh, testChecker, classExtractor))
            }
            is ITILeaf -> {
                val currentScore = scoreFunc(dt.branchLabel, dt.counts)
                if(bestScore - currentScore > scoreThresh){
                    val (passTrials, failTrials) = dt.examples.partition { testChecker(bestTest, it) }
                    val passLeaf = ITILeaf(dt.branchLabel + bestTest, createStats(vocab, passTrials, testChecker, classExtractor), passTrials, dt.testCandidates[bestTest]!!.passTrialCounts, true)
                    val failLeaf = ITILeaf(dt.branchLabel + Pair(bestTest.first, 1 - bestTest.second), createStats(vocab, passTrials, testChecker, classExtractor), failTrials, dt.testCandidates[bestTest]!!.failTrialCounts, true)
                    return ITIDecision(
                        dt.branchLabel,
                        dt.testCandidates,
                        bestTest,
                        ensureBestTest(passLeaf, vocab, scoreFunc, scoreThresh, testChecker, classExtractor),
                        ensureBestTest(failLeaf, vocab, scoreFunc, scoreThresh, testChecker, classExtractor),
                        false
                    )
                }
            }
        }
    }
    return dt
}

fun <T, C> transposeTree(dt : ITIDecision<T,C>,
                         replacementTest: RVTest,
                         vocab : Set<RandomVariable>,
                         testChecker: (RVTest, T) -> Boolean,
                         classExtractor: (T) -> C) : ITIDecision<T,C> {
    val rootTest = dt.currentTest
    if(rootTest == replacementTest) return dt

    val transposedPass = if(dt.passBranch is ITIDecision) transposeTree(dt.passBranch, replacementTest, vocab, testChecker, classExtractor) else dt.passBranch
    val transposedFail = if(dt.failBranch is ITIDecision) transposeTree(dt.failBranch, replacementTest, vocab, testChecker, classExtractor) else dt.failBranch

    if( transposedPass is ITIDecision && transposedFail is ITIDecision){
        val passRootPassReplacement = transposedPass.passBranch
        val failRootPassReplacement = transposedFail.passBranch

        val passRootFailReplacement = transposedPass.failBranch
        val failRootFailReplacement = transposedFail.failBranch

        val newPassStats = mergeTests(passRootPassReplacement, failRootPassReplacement)
        val newFailStats = mergeTests(passRootFailReplacement, failRootFailReplacement)

        val newPassBranch = ITIDecision(dt.branchLabel + replacementTest, newPassStats, rootTest, passRootPassReplacement, failRootPassReplacement, stale = true)
        val newFailBranch = ITIDecision(dt.branchLabel + Pair(replacementTest.first, 1 - replacementTest.second), newFailStats, rootTest, passRootFailReplacement, failRootFailReplacement, stale = true)
        return dt.copy(currentTest = replacementTest, passBranch = newPassBranch, failBranch = newFailBranch)
    }
    if( transposedPass is ITIDecision && transposedFail is ITILeaf) {
        return addExamples(transposedPass.copy(branchLabel = dt.branchLabel), transposedFail.examples, classExtractor, testChecker) as ITIDecision<T, C>
    }

    if(transposedFail is ITIDecision && transposedPass is ITILeaf){
        return addExamples(transposedFail.copy(branchLabel = dt.branchLabel), transposedPass.examples, classExtractor, testChecker) as ITIDecision<T, C>
    }
    if(transposedPass is ITILeaf && transposedFail is ITILeaf){
        val (newPassTrials, newFailTrials) = (transposedPass.examples + transposedFail.examples).partition { testChecker(replacementTest, it) }

        val passCounts = HashMap(dt.testCandidates[replacementTest]!!.passTrialCounts)
        val passLeafTests = createStats(vocab, newPassTrials, testChecker, classExtractor)

        val failCounts = HashMap(dt.testCandidates[replacementTest]!!.failTrialCounts)
        val failLeafTests = createStats(vocab, newFailTrials, testChecker, classExtractor)

        return dt.copy(
            currentTest = replacementTest,
            passBranch = ITILeaf(dt.branchLabel + replacementTest, passLeafTests, newPassTrials, passCounts, true),
            failBranch = ITILeaf(dt.branchLabel + Pair(replacementTest.first, 1 - replacementTest.second), failLeafTests, newFailTrials, failCounts, true),
            stale = true
        )
    }

    throw IllegalStateException("All type cases should be covered, how did you get here?")
}

fun <T, C> mergeTests(p1 : ITINode<T,C>, p2 : ITINode<T, C>) =
    mergeTests(p1.testCandidates, p2.testCandidates)

fun <T, C> createStats(testVocab : Collection<RandomVariable>, examples: List<T>, testChecker: (RVTest, T) -> Boolean, classExtractor: (T) -> C) =
    testVocab
        .flatMap { testRV -> testRV.domain.map { testVal -> Pair(Pair(testRV, testVal), createStat(testRV, testVal, examples, testChecker, classExtractor)) } }
        .associate{ it }

fun <T, C> createStat(testRV: RandomVariable, testVal : Int, examples : List<T>, testChecker: (RVTest, T) -> Boolean, classExtractor: (T) -> C) : TestStats<C> {
    val (passTrialCounts, failTrialCounts) = examples
        .partition { example -> testChecker(Pair(testRV, testVal), example) }
        .map {
            it.groupBy { classExtractor(it) }
                .mapValues { it.value.size }
                .toMutableMap()
        }
    return TestStats(testRV, testVal, passTrialCounts, failTrialCounts)
}


fun <T, C> addTrials(stat : TestStats<C>, examples: List<T>, testChecker: (RVTest, T) -> kotlin.Boolean, classExtractor: (T) -> C){
    examples.forEach {
        val passesTest = testChecker(Pair(stat.rv, stat.testVal), it)
        val classLabel = classExtractor(it)
        if(passesTest){
            val oldCount = stat.passTrialCounts[classLabel] ?: 0
            stat.passTrialCounts[classLabel] = oldCount + 1
        }
        else{
            val oldCount = stat.failTrialCounts[classLabel] ?: 0
            stat.failTrialCounts[classLabel] = oldCount + 1
        }
    }
}


fun BDeScore(rv : RandomVariable, counts : Map<Int, Int>, partialAssignment : RVAssignment, jointParamPrior: DecisionTree<Factor>, pseudoSampleSize: Double) : Double {
    val priorFactor = jointQuery(partialAssignment, jointParamPrior)
    return BDeScore(counts.toOrderedList(rv), priorFactor, pseudoSampleSize)
}

/*
fun BDeScore(rv : RandomVariable, testStat : TestStats<Int>, partialAssignment: RVAssignment, jointParamPrior: DecisionTree<Factor>, pseudoSampleSize: Double) : Double {
    val passPrior : Factor = jointQuery(partialAssignment + Pair(testStat.rv, testStat.testVal), jointParamPrior)
    val failPrior : Factor = jointQuery(partialAssignment + Pair(testStat.rv, 1 - testStat.testVal), jointParamPrior)
    return BDeScore(testStat.passTrialCounts.toOrderedList(rv), passPrior, pseudoSampleSize) + BDeScore(testStat.failTrialCounts.toOrderedList(rv), failPrior, pseudoSampleSize)
}
*/

fun BDeScore(rv : RandomVariable, counts : Map<Int, Int>, pseudoSampleSize: Double) =
    BDeScore(counts.toOrderedList(rv), pseudoSampleSize = pseudoSampleSize)

fun BDeScore(counts: List<Int>, paramPrior: Factor? = null, pseudoSampleSize: Double = 1.0) : Double{

    val betaNumeratorArgs = ArrayList<Double>()
    val betaDenominatorArgs = ArrayList<Double>()

    for((childVal, count) in counts.withIndex()){
        val pseudoCount = if(paramPrior != null) paramPrior.values[childVal] else (1.0 / counts.size) * pseudoSampleSize
        betaNumeratorArgs.add(count + pseudoCount)
        betaDenominatorArgs.add(pseudoCount)
    }

    val logBetaNumerator = logBeta(betaNumeratorArgs)
    val logBetaDenominator = logBeta(betaDenominatorArgs)

    return logBetaNumerator - logBetaDenominator
}

// Not used with prob trees : Used for parent set calculations
fun BDeuScore(rv : RandomVariable, pSet : PSet, counts : Map<RVAssignment, List<Int>>, paramPrior : Map<RVAssignment, Factor>, pseudoSampleSize: Double) : Double =
    allAssignments(pSet.toList()).sumByDouble{ pAssgn ->
        BDeScore(counts[pAssgn]!!, paramPrior[pAssgn]!!, pseudoSampleSize)
    }

fun distributeAcrossFactors(factors : List<Factor>) : Factor{
    if(!allEqual(factors.map { it.scope })){
        throw IllegalArgumentException("Scope must match exactly")
    }

    val newValues = factors
        .map { f -> f.values.map { it / factors.size } }
        .reduce { accProbs, currentProbs ->
            accProbs.zip(currentProbs).map { (acc, current) -> acc + current }
        }

    return Factor(factors[0].scope, newValues)
}

fun add(factors : List<Factor>) : Factor{
    if(!allEqual(factors.map { it.scope })){
        throw IllegalArgumentException("Scope must match exactly")
    }

    val newValues = factors
        .map { it.values }
        .reduce { accProbs, currentProbs ->
            accProbs.zip(currentProbs).map { (acc, current) -> acc + current }
        }

    return Factor(factors[0].scope, newValues)
}

fun logBeta(values: List<Double>): Double {
    val sumOfLogGammas = values.map{ Gamma.logGamma(it)}.sum()
    val logGammaOfSum = Gamma.logGamma(values.sum())
    return sumOfLogGammas - logGammaOfSum
}

fun Map<Int,Int>.toOrderedList(rv : RandomVariable) =
    rv.domain.map { this[it] ?: 0}
