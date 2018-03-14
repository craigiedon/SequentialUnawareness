import Utils.map
import Utils.merge

sealed class RewardTree{
    abstract val vocab : Set<RandomVariable>
    abstract val availableVocab : Set<RandomVariable>
    abstract val testCandidates : Map<RVTest, TestStats<Double>>
}
typealias RVTest = Pair<RandomVariable, Int>
data class RDecision(
    override val vocab: Set<RandomVariable>,
    override val availableVocab: Set<RandomVariable>,
    override val testCandidates: Map<RVTest, TestStats<Double>>,
    val bestTest: RVTest,
    val passBranch: RewardTree,
    val failBranch: RewardTree,
    val stale: Boolean = false) : RewardTree()

data class RLeaf(
    override val vocab: Set<RandomVariable>,
    override val availableVocab: Set<RandomVariable>,
    override val testCandidates: Map<RVTest, TestStats<Double>>,
    val trials: List<Trial>) : RewardTree()

data class TestStats<C>(val rv : RandomVariable, val testVal : Int, val passTrialCounts : MutableMap<C, Int>, val failTrialCounts : MutableMap<C, Int>)


fun <T> addTrial(stat : TestStats<T>, assignment : RVAssignment, classLabel : T){
    // This should also deal with missing values, in that missing data should not pass equality test
    if(assignment[stat.rv] == stat.testVal){
        val oldCount = stat.passTrialCounts[classLabel] ?: 0
        stat.passTrialCounts[classLabel] = oldCount + 1
    }
    else{
        val oldCount = stat.failTrialCounts[classLabel] ?: 0
        stat.failTrialCounts[classLabel] = oldCount + 1
    }
}

fun addTrials(stat : TestStats<Reward>, trials : List<Trial>){
    trials.forEach { addTrial(stat, it.assignment, it.reward) }
}

fun <T> removeTrial(stat : TestStats<T>, assignment: RVAssignment, classLabel: T){
    if(assignment[stat.rv] == stat.testVal){
        val oldCount = stat.passTrialCounts[classLabel] ?: throw IllegalArgumentException("Removing count, but there dont appear to be any!")
        stat.passTrialCounts[classLabel] = oldCount - 1
    }
    else{
        val oldCount = stat.failTrialCounts[classLabel] ?: throw IllegalArgumentException("Removing count, but there dont appear to be any!")
        stat.failTrialCounts[classLabel] = oldCount - 1
    }
}

fun createStat(rv : RandomVariable, testVal : Int, trials : List<Trial>) : TestStats<Reward> {
    val (passTrialCounts, failTrialCounts) = trials
        .partition { it.assignment[rv] == testVal }
        .map {
            it.groupBy { it.reward }
                .mapValues { it.value.size }
                .toMutableMap()
        }
    return TestStats(rv, testVal, passTrialCounts, failTrialCounts)
}

fun createStats(testVocab : Collection<RandomVariable>, trials: List<Trial>) =
    testVocab
        .flatMap { rv -> rv.domain.map { testVal -> Pair(Pair(rv, testVal), createStat(rv, testVal, trials)) } }
        .associate{ it }

fun <T> bestTest(tests : Map<RVTest, TestStats<T>>, allowedVocab : Set<RandomVariable>, scoreFunc : (TestStats<T>) -> Double) =
    tests.keys
        .filter { it.first in allowedVocab }
        .maxBy{scoreFunc(tests[it]!!)}!!

fun incrementalUpdate(dt : RewardTree, trial : Trial) : RewardTree {
    val exampleAddedTree = when(dt) {
        is RLeaf -> addExamples(dt, listOf(trial))
        is RDecision -> addExamples(dt, listOf(trial))
    }
    val updatedTestTree = ensureBestTest(exampleAddedTree)
    return updatedTestTree
}

fun ensureBestTest(dt: RewardTree) : RewardTree {
    if(dt is RDecision && dt.stale){
        val bestTest = bestTest(dt.testCandidates, dt.availableVocab, ::postSplitEntropy)
        val revisedNode = if(bestTest != dt.bestTest) transposeTree(dt, bestTest) else dt
        return revisedNode.copy(stale = false, passBranch = ensureBestTest(revisedNode.passBranch), failBranch = ensureBestTest(revisedNode.failBranch))
    }

    return dt
}

fun transposeTree(dt : RDecision, replacementTest: RVTest) : RDecision {
    val rootTest = dt.bestTest
    if(rootTest == replacementTest) return dt

    val transposedPass =
        if(dt.passBranch is RDecision && dt.passBranch.bestTest != replacementTest) transposeTree(dt.passBranch, replacementTest) else dt.passBranch

    val transposedFail =
        if(dt.failBranch is RDecision && dt.failBranch.bestTest != replacementTest) transposeTree(dt.failBranch, replacementTest) else dt.failBranch

    if( transposedPass is RDecision && transposedFail is RDecision){
        val passRootPassReplacement = transposedPass.passBranch
        val failRootPassReplacement = transposedFail.passBranch

        val passRootFailReplacement = transposedPass.failBranch
        val failRootFailReplacement = transposedFail.failBranch

        val newPassStats = mergeTests(passRootPassReplacement, failRootPassReplacement)
        val newFailStats = mergeTests(passRootFailReplacement, failRootFailReplacement)

        val stillAvailable = dt.availableVocab - replacementTest.first
        val newPassBranch = RDecision(dt.vocab, stillAvailable, newPassStats, rootTest, passRootPassReplacement, failRootPassReplacement, stale = true)
        val newFailBranch = RDecision(dt.vocab, stillAvailable, newFailStats, rootTest, passRootFailReplacement, failRootFailReplacement, stale = true)
        return dt.copy(bestTest = replacementTest, passBranch = newPassBranch, failBranch = newFailBranch)
    }
    if( transposedPass is RDecision && transposedFail is RLeaf) {
        return addExamples(expandAllowedVocab(transposedPass, rootTest.first), transposedFail.trials) as RDecision
    }

    if(transposedFail is RDecision && transposedPass is RLeaf){
        return addExamples(expandAllowedVocab(transposedFail, rootTest.first), transposedPass.trials) as RDecision
    }

    if(transposedPass is RLeaf && transposedFail is RLeaf){
        val (newPassTrials, newFailTrials) = (transposedPass.trials + transposedFail.trials).partition { passesTest(replacementTest, it.assignment) }
        val stillAvailable = transposedPass.availableVocab - replacementTest.first + rootTest.first
        return RDecision(
            dt.vocab,
            dt.availableVocab,
            dt.testCandidates,
            replacementTest,
            RLeaf(dt.vocab, stillAvailable, createStats(dt.vocab, newPassTrials), newPassTrials),
            RLeaf(dt.vocab, stillAvailable, createStats(dt.vocab, newFailTrials), newFailTrials),
            stale = true
        )
    }

    throw IllegalStateException("All type cases should be covered, how did you get here?")
}

fun addAdditionalVocab(dt : RewardTree, newRVs: Set<RandomVariable>) : RewardTree{
    if(dt.vocab.containsAll(newRVs)){
        throw IllegalArgumentException("These variables would not expand existing vocab")
    }
    when(dt){
        is RLeaf -> {
            val additionalTests = createStats(newRVs, dt.trials)
            val updatedTests = dt.testCandidates + additionalTests
            return RLeaf(dt.vocab + newRVs, dt.availableVocab + newRVs, updatedTests, dt.trials)
        }
        is RDecision -> {
            val newPassBranch = addAdditionalVocab(dt.passBranch, newRVs)
            val newFailBranch = addAdditionalVocab(dt.failBranch, newRVs)
            val extraPassTests = newPassBranch.testCandidates.filterKeys{ it.first in newRVs }
            val extraFailTests = newFailBranch.testCandidates.filterKeys{ it.first in newRVs }
            val extraTests = mergeTests(extraPassTests, extraFailTests)
            val updatedTests = dt.testCandidates + extraTests
            return RDecision(dt.vocab + newRVs, dt.availableVocab + newRVs, updatedTests, dt.bestTest, newPassBranch, newFailBranch, true)
        }
    }
}

@Suppress("UNCHECKED_CAST")
fun <T : RewardTree> expandAllowedVocab(dt : T, extraVocab : RandomVariable) : T{
    return when(dt){
        is RLeaf -> dt.copy(availableVocab = dt.availableVocab + extraVocab) as T
        is RDecision -> dt.copy(
            availableVocab = dt.availableVocab + extraVocab,
            passBranch = expandAllowedVocab(dt.passBranch, extraVocab),
            failBranch = expandAllowedVocab(dt.failBranch, extraVocab)
        ) as T
        else -> throw IllegalStateException("Should be impossible for T to be something else")
    }
}

fun mergeTests(t1 : RewardTree, t2: RewardTree) : Map<RVTest, TestStats<Reward>>{
    val t1Stats = when(t1){
        is RLeaf -> createStats(t1.vocab, t1.trials)
        is RDecision -> t1.testCandidates
    }

    val t2Stats = when(t2) {
        is RLeaf -> createStats(t2.vocab, t2.trials)
        is RDecision -> t2.testCandidates
    }

    return mergeTests(t1Stats, t2Stats)
}

fun <T> mergeTests(t1Stats: Map<RVTest, TestStats<T>>, t2Stats: Map<RVTest, TestStats<T>>) =
    merge(t1Stats, t2Stats, { (rv1, tVal1, passCounts1, failCounts1), (_, _, passCounts2, failCounts2) ->
        TestStats(rv1, tVal1,
            merge(passCounts1, passCounts2, Int::plus),
            merge(failCounts1, failCounts2, Int::plus))
    })

/*
fun matchingTest(nodeTest: TestStats, test : Pair<RandomVariable, Int>) =
    nodeTest.rv == test.first && nodeTest.testVal == test.second
*/

fun addExamples(dt : RewardTree, trials: List<Trial>) : RewardTree {
    if(trials.isEmpty()){
        return dt
    }

    dt.testCandidates.forEach { (_, stat) -> addTrials(stat, trials) }

    when(dt){
        is RLeaf -> {
            val newLeaf = dt.copy(trials = dt.trials + trials)

            if(!splitRequired(newLeaf)){
                return newLeaf
            }

            val bestTest = bestTest(newLeaf.testCandidates, newLeaf.availableVocab, ::postSplitEntropy)

            val (passTrials, failTrials) = newLeaf.trials.partition {passesTest(bestTest, it.assignment)}
            val stillAvailableVocab = newLeaf.availableVocab - bestTest.first

            return RDecision(dt.vocab, newLeaf.availableVocab,
                newLeaf.testCandidates,
                bestTest,
                addExamples(RLeaf(newLeaf.vocab, stillAvailableVocab, createStats(newLeaf.vocab, emptyList()), emptyList()), passTrials),
                addExamples(RLeaf(newLeaf.vocab, stillAvailableVocab, createStats(newLeaf.vocab, emptyList()), emptyList()), failTrials),
                false)
        }
        is RDecision -> {
            val (passTrials, failTrials) = trials.partition { passesTest(dt.bestTest, it.assignment) }
            val newPass = when(dt.passBranch){
                is RLeaf -> addExamples(dt.passBranch, passTrials)
                is RDecision -> addExamples(dt.passBranch, passTrials)
            }
            val newFail = when(dt.failBranch){
                is RLeaf -> addExamples(dt.failBranch, failTrials)
                is RDecision -> addExamples(dt.failBranch, failTrials)
            }

            return dt.copy(passBranch = newPass, failBranch = newFail, stale = true)
        }
    }
}

fun removeExample(dt : RewardTree, trial : Trial) : RewardTree {
    dt.testCandidates.forEach { _, stat -> removeTrial(stat, trial.assignment, trial.reward) }

    when(dt){
        is RLeaf -> {
            if(trial !in dt.trials){
                IllegalArgumentException("Trying to remove trial which is not present in tree")
            }
            val newTrials = dt.trials.filter { it != trial }
            return dt.copy(trials = newTrials)
        }
        is RDecision -> {
            // If trial passes test, it will be in the pass branch, if it fails, it will be in the fail branch
            if(passesTest(dt.bestTest, trial.assignment)){
                return dt.copy(passBranch = removeExample(dt.passBranch, trial), stale = true)
            }

            return dt.copy(failBranch = removeExample(dt.failBranch, trial), stale = true)
        }
    }
}

fun splitRequired(leaf : RLeaf) : Boolean{
    if (leaf.trials.isEmpty()) {
        return false
    }

    if(leaf.trials.any { it.reward != leaf.trials[0].reward }) {
        return true
    }

    return false
}

fun postSplitEntropy(test : TestStats<Reward>) =
    postSplitEntropy(test.passTrialCounts, test.failTrialCounts)

fun postSplitEntropy(passCounts: Map<Reward, Int>, failCounts: Map<Reward, Int>) : Double {
    val passTotal = passCounts.values.sum()
    val failTotal = failCounts.values.sum()
    val totalTrials = passTotal + failTotal

    val passEntropy = entropy(passTotal, passCounts.values)
    val failEntropy = entropy(failTotal, failCounts.values)
    return (passTotal / totalTrials) * passEntropy + (failTotal / totalTrials) * failEntropy
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

fun passesTest(test : RVTest, rvAssignment : RVAssignment) =
    rvAssignment.containsKey(test.first) && rvAssignment[test.first] == test.second