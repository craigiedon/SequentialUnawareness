import Utils.*
import org.apache.commons.math3.special.Gamma

data class SequentialTrial(val prevState : RVAssignment, val action : Action, val currentState : RVAssignment, val reward : Double)
typealias BranchLabel = Map<RVTest, Boolean>

data class ITIUpdateConfig<in T, C>(
    val exampleToClass : (T) -> C,
    val testChecker : (RVTest, T) -> Boolean,
    val leafScoreFunc : (BranchLabel, Map<C, Int>) -> Double,
    val splitScoreFunc : (BranchLabel , TestStats<C>) -> Double,
    val splitThresh : Double,
    val vocab : Set<RandomVariable>
)

sealed class ITINode<T, C>{
    abstract val branchLabel : BranchLabel
    abstract val testCandidates : Map<RVTest, TestStats<C>>
    abstract val stale : Boolean
}
data class ITILeaf<T, C>(
    override val branchLabel : BranchLabel,
    override val testCandidates: Map<RVTest, TestStats<C>>,
    val examples : List<T>,
    val counts : MutableMap<C, Int>,
    override val stale : Boolean
) : ITINode<T, C>()

data class ITIDecision<T, C>(
    override val branchLabel : BranchLabel,
    override val testCandidates: Map<RVTest, TestStats<C>>,
    val currentTest: RVTest,
    val passBranch : ITINode<T,C>,
    val failBranch : ITINode<T,C>,
    override val stale : Boolean
) : ITINode<T, C>()

typealias ITITree<T,C> = Pair<ITINode<T,C>, ITIUpdateConfig<T,C>>

typealias RVTest = Pair<RandomVariable, Int>
data class TestStats<C>(val rv : RandomVariable, val testVal : Int, val passTrialCounts : MutableMap<C, Int>, val failTrialCounts : MutableMap<C, Int>)

typealias ProbTree = ITITree<SequentialTrial, Int>
typealias ProbNode = ITINode<SequentialTrial, Int>
typealias PLeaf = ITILeaf<SequentialTrial, Int>
typealias PDecision = ITIDecision<SequentialTrial, Int>

fun probTreeConfig(rv : RandomVariable, vocab : Set<RandomVariable>, jointParamPrior: DecisionTree<Factor>, pseudoCountSize : Double, splitThresh : Double) =
    ITIUpdateConfig<SequentialTrial, Int>(
        exampleToClass = { (_, _, currentState) -> currentState[rv]!! },
        testChecker = { rvTest, (prevState) -> prevState[rvTest.first] == rvTest.second },
        leafScoreFunc = { branchLabel, rvCounts -> leafScore(rv, branchLabel, rvCounts.toOrderedList(rv), jointParamPrior, pseudoCountSize)},
        splitScoreFunc = { branchLabel, testStats -> splitterScore(rv, branchLabel, testStats, jointParamPrior, pseudoCountSize)},
        splitThresh = splitThresh,
        vocab = vocab
    )

typealias RewardTree = ITITree<Trial, Double>
typealias RewardNode = ITINode<Trial, Double>
typealias RLeaf = ITILeaf<Trial, Double>
typealias RDecision = ITIDecision<Trial, Double>

fun rewardTreeConfig(vocab : Set<RandomVariable>) =
    ITIUpdateConfig<Trial, Reward>(
        exampleToClass = { it.reward },
        testChecker = { rvTest, (assignment) -> assignment[rvTest.first] == rvTest.second } ,
        leafScoreFunc = { _, rvCounts -> weightedNegativeEntropy(rvCounts)},
        splitScoreFunc = { _, testStat -> weightedNegativeEntropy(testStat.passTrialCounts) + weightedNegativeEntropy(testStat.failTrialCounts) },
        splitThresh = 0.0,
        vocab = vocab
    )

fun emptyRewardTree(vocab : Set<RandomVariable>) : RewardTree {
    val config = rewardTreeConfig(vocab)
    return Pair(emptyNode(config), config)
}

fun emptyProbTree(rv : RandomVariable, vocab : Set<RandomVariable>, jointParamPrior: DecisionTree<Factor>, pseudoCountSize : Double, splitThresh : Double) : ProbTree {
    val config = probTreeConfig(rv, vocab, jointParamPrior, pseudoCountSize, splitThresh)
    return Pair(emptyNode(config), config)
}

fun <T, C> emptyNode(itiUpdateConfig: ITIUpdateConfig<T, C>) : ITINode<T, C>{
    val emptyLeaf = ITILeaf<T, C>(emptyMap(), createStats(itiUpdateConfig.vocab, emptyList(), itiUpdateConfig.testChecker, itiUpdateConfig.exampleToClass), emptyList(), HashMap<C, Int>(), true)
    return ensureBestTest(emptyLeaf, itiUpdateConfig.vocab, itiUpdateConfig)
}

fun <T, C> incrementalUpdate(dt : ITINode<T, C>, examples : List<T>, itiUpdateConfig: ITIUpdateConfig<T, C>): ITINode<T, C> {
    if(!countsInSync(dt)){
        throw IllegalStateException("Counts and examples out of sync")
    }
    val exampleAddedTree = addExamples(dt, examples, itiUpdateConfig.exampleToClass, itiUpdateConfig.testChecker)
    if(!countsInSync(exampleAddedTree)){
        throw IllegalStateException("Counts and examples out of sync")
    }
    val testUpdatedNode = ensureBestTest(exampleAddedTree, itiUpdateConfig.vocab, itiUpdateConfig)
    if(!countsInSync(testUpdatedNode)){
        throw IllegalStateException("Counts and examples out of sync")
    }

    return testUpdatedNode
}

fun <T, C> countsInSync(dt : ITINode<T,C>) : Boolean =
    when (dt){
        is ITILeaf -> dt.counts.values.sum() == dt.examples.size
        is ITIDecision -> countsInSync(dt.passBranch) && countsInSync(dt.failBranch)
    }


fun <T, C> changeAllowedVocab(itiTree : ITITree<T, C>, allowedVocab: Set<RandomVariable>) : ITITree<T, C> {
    fun changeAllowedVocabRec(node : ITINode<T,C>, allowedVocab: Set<RandomVariable>) : ITINode<T,C>{
        if(itiTree.second.vocab == allowedVocab){
            return node
        }
        val additionalVars = allowedVocab - itiTree.second.vocab
        val filteredTests = node.testCandidates.filterKeys{ it.first in allowedVocab }
        when(node){
            is ITILeaf -> {
                val additionalTests = createStats(additionalVars, node.examples, itiTree.second.testChecker, itiTree.second.exampleToClass)
                val updatedTests = filteredTests + additionalTests
                return node.copy(testCandidates = updatedTests, stale = true)
            }
            is ITIDecision -> {
                if(node.currentTest.first !in allowedVocab){
                    val oldPassStats = node.testCandidates[node.currentTest]!!.passTrialCounts
                    val oldFailStats = node.testCandidates[node.currentTest]!!.failTrialCounts
                    val leafCounts = Utils.merge(oldPassStats, oldFailStats, Int::plus)
                    val leafExamples = allExamples(node)
                    val additionalTests = createStats(additionalVars, leafExamples, itiTree.second.testChecker, itiTree.second.exampleToClass)
                    val updatedTests = filteredTests + additionalTests

                    return ITILeaf(node.branchLabel.toMap(), updatedTests, leafExamples, leafCounts, true)
                }

                val newPassBranch = changeAllowedVocabRec(node.passBranch, allowedVocab)
                val newFailBranch = changeAllowedVocabRec(node.failBranch, allowedVocab)

                if(additionalVars.isEmpty()){
                    return node.copy(testCandidates = filteredTests,  passBranch = newPassBranch, failBranch = newFailBranch)
                }

                val extraPassTests = newPassBranch.testCandidates.filterKeys { it.first in additionalVars }
                val extraFailTests = newFailBranch.testCandidates.filterKeys { it.first in additionalVars }
                val additionalTests = mergeTests(extraPassTests, extraFailTests)
                return node.copy(testCandidates = filteredTests + additionalTests,  passBranch = newPassBranch, failBranch = newFailBranch, stale = true)
            }
        }
    }
    return Pair(changeAllowedVocabRec(itiTree.first, allowedVocab), itiTree.second.copy(vocab = allowedVocab))
}

fun <T, C> allExamples(dt : ITINode<T,C>) : List<T> =
    when(dt){
        is ITILeaf -> dt.examples
        is ITIDecision -> allExamples(dt.passBranch) + allExamples(dt.failBranch)
    }

fun <T, C> addExamples(dt : ITINode<T,C>, examples : List<T>, config: ITIUpdateConfig<T,C>) =
    addExamples(dt, examples, config.exampleToClass, config.testChecker)

fun <T, C> addExamples(itiNode: ITINode<T,C>, examples: List<T>, classExtractor : (T) -> C, testChecker : (RVTest, T) -> Boolean) : ITINode<T, C> {
    if(examples.isEmpty()){
        return itiNode
    }

    when(itiNode){
        is ITILeaf -> {
            // Update Counts
            if(itiNode.counts.values.sum() != itiNode.examples.size){
                throw IllegalStateException("Number of examples are out of sync with counts")
            }
            for (example in examples) {
                val classLabel = classExtractor(example)
                val oldVal : Int = itiNode.counts.getOrDefault(classLabel, 0)
                itiNode.counts[classLabel] = oldVal + 1
            }
            itiNode.testCandidates.forEach { _, testStat -> addTrials(testStat, examples, testChecker, classExtractor) }

            if(itiNode.counts.values.sum() != itiNode.examples.size + examples.size){
                throw IllegalStateException("Number of examples are out of sync with counts")
            }
            return itiNode.copy(examples = itiNode.examples + examples, stale = true)
        }
        is ITIDecision -> {
            val (passTrials, failTrials) = examples.partition { testChecker(itiNode.currentTest, it) }
            val newPass = addExamples(itiNode.passBranch, passTrials, classExtractor, testChecker)
            if(!countsInSync(newPass)){
                throw IllegalStateException()
            }
            val newFail = addExamples(itiNode.failBranch, failTrials, classExtractor, testChecker)
            if(!countsInSync(newFail)){
                throw IllegalStateException()
            }

            val updatedDT = itiNode.copy(passBranch = newPass, failBranch = newFail, stale = true)
            if(!countsInSync(updatedDT)){
                throw IllegalStateException()
            }
            updatedDT.testCandidates.forEach { (_, stat) -> addTrials(stat, examples, testChecker, classExtractor) }
            if(!countsInSync(updatedDT)){
                throw IllegalStateException()
            }
            return  updatedDT
        }
    }
}

fun <T, C> ensureBestTest(dt: ITINode<T, C>, vocab : Set<RandomVariable>, config: ITIUpdateConfig<T, C>) : ITINode<T,C>{
    if(dt.stale && dt.testCandidates.isNotEmpty()){
        // We first want to filter out tests which cannot be chosen for impossibility reasons
        val trueRVs = dt.branchLabel
            .entries
            .filter { (rvTest, rvVal) -> rvVal }
            .map { it.key.first }

        val testScores = dt.testCandidates
            .filter { !(it.key in dt.branchLabel || it.key.first in trueRVs)}
            .mapValues{ (_, testStat) -> config.splitScoreFunc(dt.branchLabel, testStat) }

        if(testScores.isEmpty()){
            return dt
        }

        // If we still have potential tests remaining, then try scoring them
        val (bestTest, bestScore) = testScores.entries.maxBy { it.value }!!

        when(dt){
            is ITIDecision -> {
                // Note here: Still don't have the case where the empty test becomes best again...
                val currentTestScore = testScores[dt.currentTest]
                val shouldRevise = currentTestScore == null || (bestTest != dt.currentTest && bestScore - currentTestScore > config.splitThresh)

                val revisedNode = if(shouldRevise)
                    transposeTree(dt, bestTest, vocab, config.testChecker, config.exampleToClass)
                else dt

                if(!countsInSync(revisedNode)){
                    throw IllegalStateException()
                }

                return revisedNode.copy(stale = false,
                    passBranch = ensureBestTest(revisedNode.passBranch, vocab, config),
                    failBranch = ensureBestTest(revisedNode.failBranch, vocab, config))
            }
            is ITILeaf -> {
                val currentScore = config.leafScoreFunc(dt.branchLabel, dt.counts)
                if(bestScore - currentScore > config.splitThresh) {
                    if(dt.branchLabel.containsKey(bestTest)){
                        println("Why are we putting this test in again?")
                    }
                    val (passTrials, failTrials) = dt.examples.partition { config.testChecker(bestTest, it) }
                    val passLeaf = ITILeaf(dt.branchLabel + Pair(bestTest, true),
                        createStats(vocab, passTrials, config.testChecker, config.exampleToClass),
                        passTrials,
                        dt.testCandidates[bestTest]!!.passTrialCounts.toMutableMap(),
                        true
                    )
                    val failLeaf = ITILeaf(dt.branchLabel + Pair(bestTest, false),
                        createStats(vocab, failTrials, config.testChecker, config.exampleToClass),
                        failTrials,
                        dt.testCandidates[bestTest]!!.failTrialCounts.toMutableMap(),
                        true
                    )

                    val revisedNode = ITIDecision(
                        dt.branchLabel,
                        dt.testCandidates,
                        bestTest,
                        ensureBestTest(passLeaf, vocab, config),
                        ensureBestTest(failLeaf, vocab, config),
                        false
                    )

                    if(!countsInSync(revisedNode)){
                        throw IllegalStateException()
                    }

                    return revisedNode
                }
                else{
                    return dt.copy(stale = false)
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

        val newPassBranch = ITIDecision(dt.branchLabel + Pair(replacementTest, true), newPassStats, rootTest, passRootPassReplacement, failRootPassReplacement, stale = true)
        val newFailBranch = ITIDecision(dt.branchLabel + Pair(replacementTest, false), newFailStats, rootTest, passRootFailReplacement, failRootFailReplacement, stale = true)
        return dt.copy(currentTest = replacementTest, passBranch = newPassBranch, failBranch = newFailBranch)
    }
    if( transposedPass is ITIDecision && transposedFail is ITILeaf) {

        return addExamples(removeLabel(transposedPass, rootTest), transposedFail.examples, classExtractor, testChecker) as ITIDecision<T, C>
    }

    if(transposedFail is ITIDecision && transposedPass is ITILeaf){
        return addExamples(removeLabel(transposedFail, rootTest), transposedPass.examples, classExtractor, testChecker) as ITIDecision<T, C>
    }
    if(transposedPass is ITILeaf && transposedFail is ITILeaf){
        val (newPassTrials, newFailTrials) = (transposedPass.examples + transposedFail.examples).partition { testChecker(replacementTest, it) }

        val passCounts = HashMap(dt.testCandidates[replacementTest]!!.passTrialCounts)
        val passLeafTests = createStats(vocab, newPassTrials, testChecker, classExtractor)

        val failCounts = HashMap(dt.testCandidates[replacementTest]!!.failTrialCounts)
        val failLeafTests = createStats(vocab, newFailTrials, testChecker, classExtractor)

        return dt.copy(
            currentTest = replacementTest,
            passBranch = ITILeaf(dt.branchLabel + Pair(replacementTest, true), passLeafTests, newPassTrials, passCounts, true),
            failBranch = ITILeaf(dt.branchLabel + Pair(replacementTest, false), failLeafTests, newFailTrials, failCounts, true),
            stale = true
        )
    }

    throw IllegalStateException("All type cases should be covered, how did you get here?")
}

fun <T, C> removeLabel(itiNode: ITINode<T, C>, testVar : RVTest) : ITINode<T, C> =
    when(itiNode){
        is ITILeaf -> itiNode.copy(branchLabel = itiNode.branchLabel - testVar)
        is ITIDecision -> itiNode.copy(branchLabel = itiNode.branchLabel - testVar, passBranch = removeLabel(itiNode.passBranch, testVar), failBranch = removeLabel(itiNode.failBranch, testVar))
    }

fun <T, C> mergeTests(p1 : ITINode<T,C>, p2 : ITINode<T, C>) =
    mergeTests(p1.testCandidates, p2.testCandidates)

fun <T> mergeTests(t1Stats: Map<RVTest, TestStats<T>>, t2Stats: Map<RVTest, TestStats<T>>) =
    merge(t1Stats, t2Stats) { (rv1, tVal1, passCounts1, failCounts1), (_, _, passCounts2, failCounts2) ->
        TestStats(rv1, tVal1,
            merge(passCounts1, passCounts2, Int::plus),
            merge(failCounts1, failCounts2, Int::plus))
    }

fun <T, C> createStats(examples : List<T>, itiUpdateConfig: ITIUpdateConfig<T, C>) =
    createStats(itiUpdateConfig.vocab, examples, itiUpdateConfig.testChecker, itiUpdateConfig.exampleToClass)

fun <T, C> createStats(testVocab : Collection<RandomVariable>, examples: List<T>, testChecker: (RVTest, T) -> Boolean, classExtractor: (T) -> C) =
    testVocab
        .flatMap { testRV -> testRV.domain.indices.map { testVal -> Pair(Pair(testRV, testVal), createStat(testRV, testVal, examples, testChecker, classExtractor)) } }
        .associate{ it }

fun <T, C> createStat(testRV: RandomVariable, testVal : Int, examples : List<T>, testChecker: (RVTest, T) -> Boolean, classExtractor: (T) -> C) : TestStats<C> {
    val (passTrialCounts, failTrialCounts) = examples
        .partition { example -> testChecker(Pair(testRV, testVal), example) }
        .map { it
            .groupBy { classExtractor(it) }
            .mapValues { it.value.size }
            .toMutableMap()
        }
    return TestStats(testRV, testVal, passTrialCounts, failTrialCounts)
}

fun <T, C> classCounts(examples : List<T>, exampleToClass : (T) -> C) =
    examples
        .groupBy(exampleToClass)
        .mapValues { it.value.size }
        .toMutableMap()


fun <T, C> addTrials(stat : TestStats<C>, examples: List<T>, testChecker: (RVTest, T) -> Boolean, classExtractor: (T) -> C){
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

fun BDsScore(rv : RandomVariable, pSet : PSet, countTable : SequentialCountTable, pseudoSampleSize: Double) : Double{
    //val nonZeroParentAssignments = countTable.counts.count{ margCounts -> margCounts.any { it > 0 } }
    val nonZeroAssignments = allAssignments(pSet.toList())
        .asSequence()
        .filter { pAssgn -> countTable.getMarginalCount(pAssgn) > 0 }

    if(nonZeroAssignments.count() == 0) return Math.log(1.0)
    val priorFactor = Factor(listOf(rv), repeatNum(1.0 / (nonZeroAssignments.count() * rv.domainSize), rv.domainSize))


    val pAssignScores = nonZeroAssignments
        .map{ pAssgn ->
            BDePAssgn(countTable.getConditionalCounts(pAssgn), priorFactor, pseudoSampleSize)
        }
    return pAssignScores.sumByDouble { it }
}

fun BDsScore(rv : RandomVariable, pSet : PSet, counts : Map<RVAssignment, List<Int>>, pseudoSampleSize: Double) : Double {
    val nonZeroParentAssignments = counts.values.count{ margCounts -> margCounts.any { it > 0 } }
    if(nonZeroParentAssignments == 0) return Math.log(1.0)
    val priorFactor = Factor(listOf(rv), repeatNum(1.0 / (nonZeroParentAssignments * rv.domainSize), rv.domainSize))


    val pAssignScores = allAssignments(pSet.toList())
        .asSequence()
        .filter { pAssgn -> counts[pAssgn]!!.any { it > 0 } }
        .map{ pAssgn ->
            BDePAssgn(counts[pAssgn]!!, priorFactor, pseudoSampleSize)
        }
    return pAssignScores.sumByDouble { it }
}

fun leafScore(rv : RandomVariable, branchLabel: BranchLabel, counts : List<Int>, jointParamPrior : DecisionTree<Factor>, pseudoCountSize: Double) : Double{
    val priorCounts = scale(jointQuery(branchLabel, jointParamPrior), pseudoCountSize)
    val totalCounts = counts.zip(priorCounts.values).map { (a,b) -> a + b}
    if(totalCounts.sum() == 0.0){
        throw IllegalArgumentException("Leaf node for $rv doesn't have any counts ($counts) or even pseudo counts ($priorCounts)!")
    }

    val uniformPrior = Factor(listOf(rv), repeatNum(1.0 / rv.domainSize, rv.domainSize))

    return BDePAssgn(totalCounts, uniformPrior)
}

fun splitterScore(rv : RandomVariable, branchLabel : BranchLabel, testStats : TestStats<Int>, jointParamPrior: DecisionTree<Factor>, pseudoCountSize : Double) : Double{
    val rvTest = Pair(testStats.rv, testStats.testVal)

    val passCounts = testStats.passTrialCounts.toOrderedList(rv)
    val failCounts = testStats.failTrialCounts.toOrderedList(rv)

    val passPrior = scale(jointQuery(branchLabel + Pair(rvTest, true), jointParamPrior), pseudoCountSize)
    val failPrior = scale(jointQuery(branchLabel + Pair(rvTest, true), jointParamPrior), pseudoCountSize)

    val passTotal = passCounts.zip(passPrior.values).map { (a,b) -> a + b }
    val failTotal = failCounts.zip(failPrior.values).map { (a,b) -> a + b }

    if(passTotal.sum() == 0.0 || failTotal.sum() == 0.0){
        return Math.log(0.0)
    }

    val uniformPrior = Factor(listOf(rv), repeatNum(1.0 / rv.domainSize, rv.domainSize))
    return BDePAssgn(passTotal, uniformPrior) + BDePAssgn(failTotal, uniformPrior)
}

fun BDsScore(rv : RandomVariable, splitCounts : List<List<Int>>, pseudoSampleSize: Double) : Double{
    val nonZeroSplits = splitCounts.count{ counts -> counts.any{it > 0 } }
    if(nonZeroSplits == 0) return Math.log(1.0)

    val priorFactor = Factor(listOf(rv), repeatNum(1.0 / nonZeroSplits * rv.domainSize, rv.domainSize))

    return splitCounts
        .asSequence()
        .filter { counts -> counts.any { it > 0 } }
        .sumByDouble {  counts -> BDePAssgn(counts, priorFactor, pseudoSampleSize)
    }
}

fun <T : Number> BDePAssgn(counts: List<T>, paramPrior: Factor, pseudoSampleSize: Double = 1.0) : Double{

    val betaNumeratorArgs = ArrayList<Double>()
    val betaDenominatorArgs = ArrayList<Double>()

    for((childVal, count) in counts.withIndex()){
        val pseudoCount = paramPrior.values[childVal] * pseudoSampleSize

        betaNumeratorArgs.add(count.toDouble() + pseudoCount)
        betaDenominatorArgs.add(pseudoCount)
    }

    val logBetaNumerator = logBeta(betaNumeratorArgs)
    val logBetaDenominator = logBeta(betaDenominatorArgs)

    return logBetaNumerator - logBetaDenominator
}

fun BDeUnnormed(rv : RandomVariable, pSet : PSet, countTable : SequentialCountTable, paramPrior : Factor, pseudoCountSize: Double) : Double {
    return allAssignments(pSet.toList())
        .sumByDouble{ pAssgn ->
            BDePAssgnUnnormed(countTable.getConditionalCounts(pAssgn), paramPrior, pseudoCountSize)
        }
}

fun <T : Number> BDePAssgnUnnormed(counts : List<T>, paramPrior : Factor, pseudoSampleSize : Double) : Double{
    val betaNumeratorArgs = ArrayList<Double>()

    for((childVal, count) in counts.withIndex()){
        val pseudoCount = paramPrior.values[childVal] * pseudoSampleSize
        betaNumeratorArgs.add(count.toDouble() + pseudoCount)
    }

    return logBeta(betaNumeratorArgs)
}


fun add(factors : List<Factor>) : Factor{
    if(!allEqual(factors.map { it.scope })){
        throw IllegalArgumentException("Scope must match exactly")
    }

    val newValues = factors
        .asSequence()
        .map { it.values }
        .reduce { accProbs, currentProbs ->
            accProbs
                .zip(currentProbs)
                .map { (acc, current) -> acc + current }
        }

    return Factor(factors[0].scope, newValues)
}

fun logBeta(values: List<Double>): Double {
    val sumOfLogGammas = values.map{ Gamma.logGamma(it)}.sum()
    val logGammaOfSum = Gamma.logGamma(values.sum())
    return sumOfLogGammas - logGammaOfSum
}

fun Map<Int,Int>.toOrderedList(rv : RandomVariable) =
    rv.domain.indices.map { this[it] ?: 0}

fun weightedNegativeEntropy(counts : Map<Reward, Int>) : Double{
    val total = counts.values.sum()
    return -total * entropy(total, counts.values)
}

fun entropy(total : Int, splits : Collection<Int>) : Double{
    if(total != splits.sum()){
        throw IllegalArgumentException("Number of items in splits does not equal total")
    }

    return - splits.sumByDouble {
        val proportion = it.toDouble() / total
        proportion * Math.log(proportion)
    }
}

fun <T, C> removeTrial(stat : TestStats<C>, example: T, testChecker: (RVTest, T) -> Boolean, classExtractor: (T) -> C){
    val rvTest = Pair(stat.rv, stat.testVal)
    val classLabel = classExtractor(example)
    if(testChecker(rvTest, example)){
        val oldCount = stat.passTrialCounts[classLabel] ?: throw IllegalArgumentException("Removing count, but there dont appear to be any!")
        stat.passTrialCounts[classLabel] = oldCount - 1
    }
    else{
        val oldCount = stat.failTrialCounts[classLabel] ?: throw IllegalArgumentException("Removing count, but there dont appear to be any!")
        stat.failTrialCounts[classLabel] = oldCount - 1
    }
}

fun <T, C> removeExample(dt : ITINode<T,C>, example: T, updateConfig : ITIUpdateConfig<T,C>) : ITINode<T,C> {
    dt.testCandidates.forEach { _, stat -> removeTrial(stat, example, updateConfig.testChecker, updateConfig.exampleToClass) }

    when(dt){
        is ITILeaf -> {
            if(example !in dt.examples){
                IllegalArgumentException("Trying to remove example which is not present in tree")
            }
            val newExamples = dt.examples.filter { it != example }
            return dt.copy(examples = newExamples)
        }
        is ITIDecision -> {
            // If example passes test, it will be in the pass branch, if it fails, it will be in the fail branch
            if(updateConfig.testChecker(dt.currentTest, example)){
                return dt.copy(passBranch = removeExample(dt.passBranch, example, updateConfig), stale = true)
            }

            return dt.copy(failBranch = removeExample(dt.failBranch, example, updateConfig), stale = true)
        }
    }
}
