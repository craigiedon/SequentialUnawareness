import Utils.allEqual
import Utils.map
import org.apache.commons.math3.special.Gamma

data class SequentialTrial(val prevState : RVAssignment, val action : Action, val currentState : RVAssignment, val reward : Double)
sealed class ProbTree {
    abstract val rv : RandomVariable
    abstract val vocab : Set<RandomVariable>
    abstract val branchLabel : RVAssignment
    abstract val testCandidates : Map<RVTest, TestStats<Int>>
    abstract val jointParamPrior : DecisionTree<Factor>
    abstract val pseudoCountSize : Double
    val availableVocab get() = vocab - branchLabel.keys
}
data class PLeaf(
    override val rv: RandomVariable,
    override val vocab: Set<RandomVariable>,
    override val testCandidates: Map<RVTest, TestStats<Int>>,
    override val branchLabel : RVAssignment,
    override val jointParamPrior : DecisionTree<Factor>,
    override val pseudoCountSize : Double,
    val trials: List<SequentialTrial>,
    val counts: MutableMap<Int, Int>) : ProbTree()

data class PDecision(
    override val rv: RandomVariable,
    override val vocab: Set<RandomVariable>,
    override val testCandidates: Map<RVTest, TestStats<Int>>,
    override val branchLabel : RVAssignment,
    override val jointParamPrior : DecisionTree<Factor>,
    override val pseudoCountSize : Double,
    val currentTest: RVTest,
    val passBranch: ProbTree,
    val failBranch: ProbTree,
    val stale: Boolean) : ProbTree()


fun emptyPLeaf(rv : RandomVariable, vocab : Set<RandomVariable>, jointParamPrior: DecisionTree<Factor>, pseudoSampleSize: Double) : ProbTree {
    val leaf = PLeaf(rv, vocab, createStats(vocab, emptyList(), rv), emptyMap(), jointParamPrior, pseudoSampleSize, emptyList(), HashMap())
    return attemptSplit(leaf)
}

fun incrementalUpdate(probTree: ProbTree, seqTrials : List<SequentialTrial>) : ProbTree {
    val exampleAddedTree = addExamples(probTree, seqTrials)
    return ensureBestTest(exampleAddedTree)
}

fun addExamples(pTree: ProbTree, seqTrials: List<SequentialTrial>) : ProbTree {
    if(seqTrials.isEmpty()){
        return pTree
    }

    when(pTree){
        is PLeaf -> {
            // Update Counts
            for ((_, _, currentState) in seqTrials) {
                val classLabel = currentState[pTree.rv]!!
                val oldVal = pTree.counts.getOrDefault(classLabel, 0)
                pTree.counts[classLabel] = oldVal + 1
            }
            pTree.testCandidates.forEach { _, testStat -> addTrials(testStat, seqTrials, pTree.rv) }

            return attemptSplit(pTree)
        }
        is PDecision -> {
            val (passTrials, failTrials) = seqTrials.partition { passesTest(pTree.currentTest, it.prevState) }
            val newPass = addExamples(pTree.passBranch, passTrials)
            val newFail = addExamples(pTree.failBranch, failTrials)

            val updatedDT = pTree.copy(passBranch = newPass, failBranch = newFail, stale = true)
            updatedDT.testCandidates.forEach { (_, stat) -> addTrials(stat, seqTrials, pTree.rv) }
            return  updatedDT
        }
    }
}

fun ensureBestTest(dt: ProbTree) : ProbTree{
    if(dt is PDecision && dt.stale){
        val testScores : Map<RVTest, Double> = dt.testCandidates.mapValues{ (_, testStat) -> BDeScore(dt.rv, testStat, dt.branchLabel, dt.jointParamPrior, dt.pseudoCountSize) }

        val (bestTest, bestScore) = testScores.entries.maxBy { it.value }!!

        val revisedNode = if(bestTest != dt.currentTest && bestScore - testScores[dt.currentTest]!! > 0.7)
            transposeTree(dt, bestTest)
        else dt

        return revisedNode.copy(stale = false, passBranch = ensureBestTest(revisedNode.passBranch), failBranch = ensureBestTest(revisedNode.failBranch))
    }

    return dt
}

fun transposeTree(dt : PDecision, replacementTest: RVTest) : PDecision {
    val rootTest = dt.currentTest
    if(rootTest == replacementTest) return dt

    val transposedPass = if(dt.passBranch is PDecision) transposeTree(dt.passBranch, replacementTest) else dt.passBranch
    val transposedFail = if(dt.failBranch is PDecision) transposeTree(dt.failBranch, replacementTest) else dt.failBranch

    if( transposedPass is PDecision && transposedFail is PDecision){
        val passRootPassReplacement = transposedPass.passBranch
        val failRootPassReplacement = transposedFail.passBranch

        val passRootFailReplacement = transposedPass.failBranch
        val failRootFailReplacement = transposedFail.failBranch

        val newPassStats = mergeTests(passRootPassReplacement, failRootPassReplacement)
        val newFailStats = mergeTests(passRootFailReplacement, failRootFailReplacement)

        val newPassBranch = PDecision(dt.rv, dt.vocab, newPassStats, dt.branchLabel + replacementTest, dt.jointParamPrior, dt.pseudoCountSize, rootTest, passRootPassReplacement, failRootPassReplacement, stale = true)
        val newFailBranch = PDecision(dt.rv, dt.vocab, newFailStats, dt.branchLabel + Pair(replacementTest.first, 1 - replacementTest.second), dt.jointParamPrior, dt.pseudoCountSize, rootTest, passRootFailReplacement, failRootFailReplacement, stale = true)
        return dt.copy(currentTest = replacementTest, passBranch = newPassBranch, failBranch = newFailBranch)
    }
    if( transposedPass is PDecision && transposedFail is PLeaf) {
        return addExamples(transposedPass.copy(branchLabel = dt.branchLabel), transposedFail.trials) as PDecision
    }

    if(transposedFail is PDecision && transposedPass is PLeaf){
        return addExamples(transposedFail.copy(branchLabel = dt.branchLabel), transposedPass.trials) as PDecision
    }
    if(transposedPass is PLeaf && transposedFail is PLeaf){
        val (newPassTrials, newFailTrials) = (transposedPass.trials + transposedFail.trials).partition { passesTest(replacementTest, it.prevState) }
        val stillAvailable = transposedPass.availableVocab - replacementTest.first + rootTest.first

        val passCounts = HashMap(dt.testCandidates[replacementTest]!!.passTrialCounts)
        val passLeafTests = createStats(dt.vocab, newPassTrials, dt.rv)

        val failCounts = HashMap(dt.testCandidates[replacementTest]!!.failTrialCounts)
        val failLeafTests = createStats(dt.vocab, newFailTrials, dt.rv)

        return dt.copy(
            currentTest = replacementTest,
            passBranch = PLeaf(dt.rv, dt.vocab, passLeafTests, dt.branchLabel + replacementTest, dt.jointParamPrior, dt.pseudoCountSize, newPassTrials, passCounts),
            failBranch = PLeaf(dt.rv, dt.vocab, failLeafTests, dt.branchLabel + Pair(replacementTest.first, 1 - replacementTest.second), dt.jointParamPrior, dt.pseudoCountSize, newFailTrials, failCounts),
            stale = true
        )
    }

    throw IllegalStateException("All type cases should be covered, how did you get here?")
}

fun changeVocab(dt : ProbTree, newVocab: Set<RandomVariable>) : ProbTree{
    if(dt.vocab == newVocab){
        return dt
    }
    val additionalVars = newVocab - dt.vocab
    val filteredTests = dt.testCandidates.filterKeys{ it.first in newVocab }
    when(dt){
        is PLeaf -> {
            val additionalTests = createStats(additionalVars, dt.trials, dt.rv)
            val updatedTests = filteredTests + additionalTests
            return PLeaf(dt.rv, newVocab, updatedTests, dt.branchLabel, dt.jointParamPrior, dt.pseudoCountSize, dt.trials, dt.counts)
        }
        is PDecision -> {
            val newPassBranch = changeVocab(dt.passBranch, newVocab)
            val newFailBranch = changeVocab(dt.failBranch, newVocab)
            if(additionalVars.isEmpty() && dt.currentTest.first in newVocab){
                return PDecision(dt.rv, newVocab, filteredTests, dt.branchLabel, dt.jointParamPrior, dt.pseudoCountSize, dt.currentTest, newPassBranch, newFailBranch, dt.stale)
            }
            val extraPassTests = newPassBranch.testCandidates.filterKeys { it.first in additionalVars }
            val extraFailTests = newFailBranch.testCandidates.filterKeys { it.first in additionalVars }
            val additionalTests = mergeTests(extraPassTests, extraFailTests)
            return PDecision(dt.rv, newVocab, filteredTests + additionalTests, dt.branchLabel, dt.jointParamPrior, dt.pseudoCountSize, dt.currentTest, newPassBranch, newFailBranch, true)
        }
    }
}

fun mergeTests(p1 : ProbTree, p2 : ProbTree) : Map<RVTest, TestStats<Int>> {
    val p1Tests = when(p1){
        is PLeaf -> p1.testCandidates
        is PDecision -> p1.testCandidates
    }

    val p2Tests = when(p2){
        is PLeaf -> p2.testCandidates
        is PDecision -> p2.testCandidates
    }
    return mergeTests(p1Tests, p2Tests)
}

fun createStats(testVocab : Collection<RandomVariable>, seqTrials: List<SequentialTrial>, rv : RandomVariable) =
    testVocab
        .flatMap { testRV -> testRV.domain.map { testVal -> Pair(Pair(testRV, testVal), createStat(testRV, testVal, seqTrials, rv)) } }
        .associate{ it }

fun createStat(testRV: RandomVariable, testVal : Int, trials : List<SequentialTrial>, rv : RandomVariable) : TestStats<Int> {
    val (passTrialCounts, failTrialCounts) = trials
        .partition { it.prevState[testRV] == testVal }
        .map {
            it.groupBy { it.currentState[rv]!! }
                .mapValues { it.value.size }
                .toMutableMap()
        }
    return TestStats(testRV, testVal, passTrialCounts, failTrialCounts)
}


fun addTrials(stat : TestStats<Int>, trials : List<SequentialTrial>, predictionRV: RandomVariable){
    trials.forEach { addTrial(stat, it.prevState, it.currentState[predictionRV]!!) }
}

fun attemptSplit(leaf : PLeaf) : ProbTree{
    val currentScore = BDeScore(leaf.rv, leaf.counts, leaf.branchLabel, leaf.jointParamPrior, leaf.pseudoCountSize)
    val scoreIncreases = leaf.testCandidates
        .mapValues { (_, tStats) ->
            val candidateScore = BDeScore(leaf.rv, tStats, leaf.branchLabel, leaf.jointParamPrior, leaf.pseudoCountSize)
            candidateScore - currentScore
        }
        .filterValues { it > 0.7 } // If split model is roughly twice as likely as leaf model (Log(2.0) ~= 0.7)

    if(scoreIncreases.isEmpty()){
        return leaf
    }

    val bestTest = scoreIncreases.maxBy { it.value }!!.key
    val (passTrials, failTrials) = leaf.trials.partition { passesTest(bestTest, it.prevState) }

    val passBranch = PLeaf(leaf.rv, leaf.vocab, createStats(leaf.vocab, emptyList(), leaf.rv), leaf.branchLabel + bestTest, leaf.jointParamPrior, leaf.pseudoCountSize, emptyList(), HashMap<Int,Int>())
    val failBranch = PLeaf(leaf.rv, leaf.vocab, createStats(leaf.vocab, emptyList(), leaf.rv), leaf.branchLabel + Pair(bestTest.first, 1 - bestTest.second), leaf.jointParamPrior, leaf.pseudoCountSize, emptyList(), HashMap<Int, Int>())

    return PDecision(leaf.rv, leaf.vocab,
        leaf.testCandidates,
        leaf.branchLabel,
        leaf.jointParamPrior,
        leaf.pseudoCountSize,
        bestTest,
        addExamples(passBranch, passTrials),
        addExamples(failBranch, failTrials),
        false)
}

fun BDeScore(rv : RandomVariable, counts : Map<Int, Int>, partialAssignment : RVAssignment, jointParamPrior: DecisionTree<Factor>, pseudoSampleSize: Double) : Double {
    val priorFactor = jointQuery(partialAssignment, jointParamPrior)
    return BDeScore(counts.toOrderedList(rv), priorFactor, pseudoSampleSize)
}

fun BDeScore(rv : RandomVariable, testStat : TestStats<Int>, partialAssignment: RVAssignment, jointParamPrior: DecisionTree<Factor>, pseudoSampleSize: Double) : Double {
    val passPrior : Factor = jointQuery(partialAssignment + Pair(testStat.rv, testStat.testVal), jointParamPrior)
    val failPrior : Factor = jointQuery(partialAssignment + Pair(testStat.rv, 1 - testStat.testVal), jointParamPrior)
    return BDeScore(testStat.passTrialCounts.toOrderedList(rv), passPrior, pseudoSampleSize) + BDeScore(testStat.failTrialCounts.toOrderedList(rv), failPrior, pseudoSampleSize)
}

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


/*
fun probFromTree(currentState : RVAssignment, rvVal: Int, probTree : ProbTree) : Double =
    when(probTree){
        is PLeaf -> probTree.counts[rvVal]!! / probTree.counts.values.sum().toDouble()
        is PDecision -> {
            val (testRV, testVal) = probTree.currentTest
            if(currentState[testRV]!! == testVal){
                probFromTree(currentState, rvVal, probTree.passBranch)
            }
            else{
                probFromTree(currentState, rvVal, probTree.failBranch)
            }
        }
    }

fun conditionalProb(currentState : RVAssignment, proposedAssignment : RVAssignment, dbn : DynamicBayesNet) =
    proposedAssignment.entries
        .fold(1.0, { prob, (rv, rvVal) ->
            val factor = matchLeaf(dbn.cpdTrees[rv]!!, currentState).value
            prob * factor.values[rvVal]
        })
*/
