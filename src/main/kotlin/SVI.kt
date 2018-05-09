import Utils.product
import Utils.productByDouble
import com.google.ortools.constraintsolver.Decision
import javax.swing.text.html.parser.DTD
import kotlin.streams.toList

sealed class DecisionTree<T>
data class DTLeaf<T>(val value: T) : DecisionTree<T>()
data class DTDecision<T>(val rvTest: RVTest, val passBranch : DecisionTree<T>, val failBranch : DecisionTree<T>) : DecisionTree<T>()

typealias NaiveFactors = Map<RandomVariable, Factor>
typealias Action = String
typealias QTree = DecisionTree<Double>
typealias VTree = DecisionTree<Double>
typealias PolicyTree = DecisionTree<Action>

fun <T> vocabInDT(dt : DecisionTree<T>) : PSet =
    when(dt){
        is DTLeaf -> emptySet()
        is DTDecision -> vocabInDT(dt.passBranch) + vocabInDT(dt.failBranch) + dt.rvTest.first
    }

fun <T> leaves(dt : DecisionTree<T>) : List<DTLeaf<T>> =
   when(dt){
       is DTLeaf<T> -> listOf(dt)
       is DTDecision -> leaves(dt.passBranch) + leaves(dt.failBranch)
   }

fun <T> leavesWithHistory(dt: DecisionTree<T>, currentHistory : Map<RVTest, Boolean> = emptyMap()) : List<Pair<Map<RVTest, Boolean>, DTLeaf<T>>> =
    when(dt){
        is DTLeaf<T> -> listOf(Pair(currentHistory, dt))
        is DTDecision<T> -> leavesWithHistory(dt.passBranch, currentHistory + Pair(dt.rvTest, true)) +
                            leavesWithHistory(dt.failBranch, currentHistory + Pair(dt.rvTest,false))
    }


fun <T, S> foldTree (dTree : DecisionTree<T>, initial : S, accFunc : (S, T) -> S) : S =
    when(dTree){
        is DTLeaf -> accFunc(initial, dTree.value)
        is DTDecision -> {
            val foldedPass = foldTree(dTree.passBranch, initial, accFunc)
            foldTree(dTree.failBranch, foldedPass, accFunc)
        }
    }

typealias DynamicBayesNet = Map<RandomVariable, DecisionTree<Factor>>

fun convertFromITI(itiTree: RewardNode) : DecisionTree<Reward> =
    when(itiTree){
        is RLeaf -> {
            val averageReward = if(itiTree.examples.isEmpty()){
                0.0
            }
            else{
                itiTree.examples.map { it.reward }.average()
            }
            DTLeaf(averageReward)

        } // This would be quicker if these stats had been pre-computed at leaf
        is RDecision -> DTDecision(
            itiTree.currentTest,
            convertFromITI(itiTree.passBranch),
            convertFromITI(itiTree.failBranch)
        )
    }

fun convertToCPT(childRV: RandomVariable, itiTree: ProbNode, jointParamPrior : DecisionTree<Factor>, pseudoCountStrength: Double) : DecisionTree<Factor> =
    when(itiTree){
        is ITILeaf -> {
            val relevantPrior = jointQuery(itiTree.branchLabel, jointParamPrior)
            val bestParams = maxPosteriorParams(childRV, itiTree.counts, relevantPrior, pseudoCountStrength)
            DTLeaf(bestParams)
        }
        is ITIDecision -> DTDecision(itiTree.currentTest,
            convertToCPT(childRV, itiTree.passBranch, jointParamPrior, pseudoCountStrength),
            convertToCPT(childRV, itiTree.failBranch, jointParamPrior, pseudoCountStrength)
        )
    }

// Take a decision tree for joint probabilities, find all leaves that match assignment, and combine them
// For leaves that make no assignment to some vars in parent assignment, assume counts would be uniformly distributed over missing context
fun jointQuery(parentAssgn : Map<RVTest, Boolean>, jointDT: DecisionTree<Factor>) : Factor{
    fun jointQueryRec(dt : DecisionTree<Factor>, matchedTests: Set<RVTest>) : Factor =
        when(dt){
            is DTLeaf -> {
                val unmatchedTests = (parentAssgn.keys - matchedTests)
                // I think the div amount might be a bit more complicated than this if the tests are on rvs with non-binary domains
                val divAmount = Math.pow(2.0, unmatchedTests.size.toDouble())
                val scaledVals = dt.value.values.map { it / divAmount }
                Factor(dt.value.scope, scaledVals)
            }
            is DTDecision -> {
                if(dt.rvTest in parentAssgn){
                    when(parentAssgn[dt.rvTest]!!){
                        true -> jointQueryRec(dt.passBranch, matchedTests + dt.rvTest)
                        false -> jointQueryRec(dt.failBranch, matchedTests + dt.rvTest)
                    }
                }
                else{
                    val relevantTests = parentAssgn.filterKeys { it.first == dt.rvTest.first }
                    if(relevantTests.any{ it.value }){
                        jointQueryRec(dt.failBranch, matchedTests + dt.rvTest)
                    }
                    else if(relevantTests.size == dt.rvTest.first.domainSize - 1){
                        jointQueryRec(dt.passBranch, matchedTests + dt.rvTest)
                    }
                    else{
                        add(listOf(
                            jointQueryRec(dt.passBranch, matchedTests),
                            jointQueryRec(dt.failBranch, matchedTests))
                        )
                    }
                }
            }
        }
    return jointQueryRec(jointDT, emptySet())
}

fun convertToJointProbTree(rv : RandomVariable, itiNode: ProbNode, priorJointDT: DecisionTree<Factor>, psuedoCountStrength : Double) : DecisionTree<Factor> {
    val totalTrials = when(itiNode){
        is PLeaf -> itiNode.examples.size
        is PDecision -> {
            val testStat = itiNode.testCandidates[itiNode.currentTest]!!
            testStat.passTrialCounts.values.sum() + testStat.failTrialCounts.values.sum()
        }
    }

    fun convertToJointProbTreeRec(pt : ProbNode) :DecisionTree<Factor> =
        when(pt){
            is ITILeaf -> {
                val jointPriorParam = jointQuery(pt.branchLabel, priorJointDT)
                val realCounts = rv.domain.indices.map { (pt.counts[it] ?: 0)}
                val scaledCombined = rv.domain.indices.map { (realCounts[it] + (jointPriorParam.values[it] * psuedoCountStrength)) / (totalTrials + psuedoCountStrength)}
                DTLeaf(Factor(listOf(rv), scaledCombined))
            }
            is ITIDecision -> DTDecision(pt.currentTest,
                convertToJointProbTreeRec(pt.passBranch),
                convertToJointProbTreeRec(pt.failBranch)
            )
        }
    return convertToJointProbTreeRec(itiNode)
}

fun partialMatch(partialAssignment: RVAssignment, fullAssignment : RVAssignment) =
    partialAssignment == fullAssignment.filterKeys { it in partialAssignment.keys }

fun maxPosteriorParams(rv : RandomVariable, counts : Map<Int, Int>, priorParams : Factor, pseudoCountStrength : Double) : Factor{
    val totalCounts = rv.domain.indices.sumBy { counts[it] ?: 0}
    val totalPseudoCounts = rv.domain.indices.sumByDouble { priorParams.values[it] * pseudoCountStrength }
    val bestParams = rv.domain.indices.map { rvVal -> ((counts[rvVal] ?: 0) + priorParams.values[rvVal] * pseudoCountStrength) / (totalCounts + totalPseudoCounts) }
    return Factor(listOf(rv), bestParams)
}

fun structuredValueIteration(rewardTree : DecisionTree<Double>, actionDBNs: Map<Action, DynamicBayesNet>, vocab : List<RandomVariable>, pruneRange: Double = 0.0) : Pair<PolicyTree, VTree>{
    var rangedValTree : DecisionTree<Range> = toRanged(rewardTree)
    val testOrder : List<RVTest> = vocab.flatMap { rv -> rv.domain.indices.map { RVTest(rv, it)} }
    do{
        val qTreesRanged = actionDBNs.map { (a,dbn) ->
            regressRanged(rangedValTree, rewardTree, dbn) }

        val orderedQs = qTreesRanged.map { setOrder(it, testOrder) }
        val mergedQTrees = simplify(orderedMerge(testOrder, orderedQs, ::rangeCeilings), ::doubleEquality)
        val newValueTree = reOrderDT(mergedQTrees)
        val prunedValueTree = prune(newValueTree, pruneRange)

        val diffTree = append(rangedValTree, newValueTree, ::rangeDistance, ::doubleEquality)
        val biggestDiff = foldTree(diffTree, 0.0, ::maxOf)

        rangedValTree = prunedValueTree


        println("Unordered Qs: ${qTreesRanged.map { numLeaves(it) }}")
        println("Num Vals: Before Prune : ${numLeaves(newValueTree)} After Prune : ${numLeaves(prunedValueTree)}")
        println("Biggest Diff : $biggestDiff")
        println()
    } while(biggestDiff > 0.01)

    val finalQTrees = actionDBNs.mapValues { regressRanged(rangedValTree, rewardTree, it.value) }
    val finalPolicy = choosePolicy(finalQTrees, vocab)
    return Pair(finalPolicy, fromRanged(rangedValTree))
}

fun rangeCeilings(r1 : Range, r2: Range) : Range{
    return Range(maxOf(r1.lower, r2.lower), maxOf(r1.upper, r2.upper))
}

fun rangeDistance(r1 : Range, r2 : Range) : Double{
    if(r1.upper < r2.lower){
        return r2.lower - r1.upper
    }
    if(r2.upper < r1.lower){
        return r1.lower - r2.upper
    }

    return 0.0
}

data class Range(val lower : Double, val upper : Double)

fun <T> penultimateNodes(dt: DecisionTree<T>) : List<DTDecision<T>> =
    when(dt){
        is DTLeaf -> emptyList()
        is DTDecision ->
            if(dt.passBranch is DTLeaf && dt.failBranch is DTLeaf) listOf(dt)
            else penultimateNodes(dt.passBranch) + penultimateNodes(dt.failBranch)
    }

fun incrementalSVI(rewardTree : DecisionTree<Double>,
                   valueTree : DecisionTree<Range>,
                   actionDBNs : Map<Action, DynamicBayesNet>,
                   vocab : List<RandomVariable>,
                   pruningRange: Double = 0.0) : Pair<DecisionTree<Range>, Map<Action, DecisionTree<Range>>>{

    val qTrees = actionDBNs.entries
        //.parallelStream()
        .map { (action, dbn) -> Pair(action, regressRanged (valueTree, rewardTree, dbn)) }
        .toList()
        .associate { it }

    val testOrder = vocab.flatMap { rv -> rv.domain.indices.map { RVTest(rv, it) } }
    val orderedQVals = qTrees.mapValues { setOrder(it.value, testOrder) }

    val newValueTree = reOrderDT(simplify(orderedMerge(testOrder, orderedQVals.values.toList(), ::rangeCeilings), ::doubleEquality))
    val finalValTree = prune(newValueTree, pruningRange)

    println("Q Tree Sizes: ${qTrees.map{ (a, qt) -> Pair(a, numLeaves(qt)) }}")
    println("Val tree size: Before Pruning ${numLeaves(newValueTree)} After Pruning: ${numLeaves(finalValTree)}")
    return Pair(finalValTree, qTrees)
}


fun choosePolicy(qTrees : Map<Action, DecisionTree<Range>>, vocab : List<RandomVariable>) : PolicyTree {
    val testOrder = vocab.flatMap { rv -> rv.domain.indices.map { RVTest(rv, it) } }
    val orderedQVals = qTrees.mapValues { fromRanged(setOrder(it.value, testOrder)) }


    val annotatedActionQs = orderedQVals.map {(a, qt) -> fMap(qt, {qVal -> Pair(a, qVal)})}
    val annotatedPolicy = orderedMerge(testOrder, annotatedActionQs, ::maxBySecond)
    val unsimplifiedPolicy = fMap(annotatedPolicy, {(action, _) -> action})
    val simplifiedPolicy = simplify(unsimplifiedPolicy, {a1, a2 -> a1 == a2 })
    return simplifiedPolicy
}

/*
fun reOrderPolicy(dt : DecisionTree<Action>) : DecisionTree<Action> {
    when(dt){
        is DTLeaf -> return dt
        is DTDecision -> {
            val dtTests = getAllTests(dt)
            val varActionCounts = actionsForVarSplits(dtTests, dt)
            val varEntropies = varActionCounts.mapValues { (_, actionCounts) -> intEntropy(actionCounts) }
            val (bestTest, _) = varEntropies.maxBy { it.value }!!

            val transposedTree = transposeTree(dt, bestTest)
            return when(transposedTree){
                is DTLeaf -> transposedTree
                is DTDecision -> transposedTree.copy(passBranch = reOrderPolicy(transposedTree.passBranch), failBranch = reOrderPolicy(transposedTree.failBranch))
            }
        }
    }
}
*/

/*
fun actionsForVarSplits(tests : Set<RVTest>, dt : DecisionTree<Action>) : Map<RVTest, List<Int>>{
    val actionSplits = tests
        .associate { rvTest -> Pair(rvTest, listOf(false, true)
            .map { emptySet<Action>() }
            .toMutableList())
        }
        .toMutableMap()

    fun splitsForVarsRec(node : DecisionTree<Action>, branchLabel : Map<RVTest, Boolean>){
        when(node){
            is DTLeaf -> {
                for(rvTest in tests){
                    val rvSplit = actionSplits[rvTest]!!
                    if(branchLabel.containsKey(rvTest)){
                        val testLabel = if(branchLabel[rvTest]!!) 1 else 0
                        rvSplit[testLabel] = rvSplit[testLabel] + node.value
                    }
                    else{
                        actionSplits[rvTest] = rvSplit.map { it + node.value }.toMutableList()
                    }
                }
            }
            is DTDecision -> {
                splitsForVarsRec(node.passBranch, branchLabel + Pair(node.rvTest, true))
                splitsForVarsRec(node.failBranch, branchLabel + Pair(node.rvTest, false))
            }
        }
    }

    splitsForVarsRec(dt, emptyMap())
    return actionSplits.mapValues { it.value.map { it.size } }
}
*/

fun <T, S : Comparable<S>> maxBySecond(p1 : Pair<T,S>, p2 : Pair<T, S>) =
    if(p1.second >= p2.second) p1 else p2


fun regressRanged(rangedValTree : DecisionTree<Range>, rewardTree : DecisionTree<Double>, dbn: DynamicBayesNet) : DecisionTree<Range> {
    val discountFactor = 0.9

    val pTree = pRegress(rangedValTree, dbn)
    val fvTree = undiscountedFutureValueRanged(rangedValTree, pTree)

    val rewardRange = toRanged(rewardTree)
    val newTree = append(fvTree, rewardRange,
        { (vLower, vUpper), (rLower, rUpper) -> Range(discountFactor * vLower + rLower, discountFactor * vUpper + rUpper)},
        ::doubleEquality)
    return newTree
}

fun regress(valTree : DecisionTree<Double>, rewardTree : DecisionTree<Double>, dbn : DynamicBayesNet) :  QTree {
    val discountFactor = 0.9

    val pTree = pRegress(valTree, dbn)

    val fvTree = undiscountedFutureValue(valTree, pTree)
    val newTree = append(fvTree, rewardTree, {v, r -> discountFactor * v + r}, ::doubleEquality)
    /*
    println("pTree Size: ${numLeaves(pTree)}")
    println("fvTree Size: ${numLeaves(fvTree)}")
    println("newTree Size: ${numLeaves(newTree)}")
    */
    return newTree
}


fun undiscountedFutureValue(valueTree : DecisionTree<Double>, pTree : DecisionTree<NaiveFactors>) : DecisionTree<Double>{
    val vTreeWithAssign = withBranchAssignments(valueTree)
    val vLeaves = leaves(vTreeWithAssign)
    val undiscountedVals = fMap(pTree, { pLeaf ->
        vLeaves.parallelStream()
            .map { vLeaf -> vLeaf.value.second * probability(vLeaf.value.first, pLeaf) }
            .reduce(0.0, { d1, d2 -> d1 + d2  })
    })
    return undiscountedVals
}

fun undiscountedFutureValueRanged(rangedValTree : DecisionTree<Range>, pTree : DecisionTree<NaiveFactors>) : DecisionTree<Range>{
    val vTreeWithLabels = disjAssignmentAug(rangedValTree)

    return fMap(pTree, {pLeafFactors ->
        weightedValueSum(vTreeWithLabels, pLeafFactors)
    })
}

fun weightedValueSum(vt: AugDT<RVAssignmentDisj, Range>, naiveFactors : NaiveFactors) : Range{
   when(vt){
       is AugLeaf -> {
           val label = vt.augInfo
           val rawRange = vt.leaf.value
           val prob = probability(label, naiveFactors)
           return Range(rawRange.lower * prob, rawRange.upper * prob)
       }
       is AugDecision -> {
           val rv = vt.rvTest.first
           val passAssgn = vt.passBranch.augInfo[rv]!!
           val failAssgn = vt.failBranch.augInfo[rv]!!

           val passPartProb = naiveFactors[rv]!!.values.filterIndexed { i, _ -> i in passAssgn }.sum()
           val failPartProb = naiveFactors[rv]!!.values.filterIndexed { i, _ -> i in failAssgn }.sum()
           if(passPartProb > 0.0 && failPartProb > 0.0){
               return add(
                   weightedValueSum(vt.passBranch, naiveFactors),
                   weightedValueSum(vt.failBranch, naiveFactors)
               )
           }
           else if(passPartProb > 0.0){
               return weightedValueSum(vt.passBranch, naiveFactors)
           }
           else if(failPartProb > 0.0){
               return weightedValueSum(vt.failBranch, naiveFactors)
           }
           else{
               throw IllegalStateException("Neither the pass or fail branches appear to be possible. How did you get here?")
           }
       }
   }
}

fun add(r1 : Range, r2 : Range) = Range(r1.lower + r2.lower, r1.upper + r2.upper)

typealias RVAssignmentDisj = Map<RandomVariable, Set<Int>>
fun <T> withBranchAssignments(dt : DecisionTree<T>) : DecisionTree<Pair<RVAssignmentDisj, T>> {
    fun withBranchRec(node : DecisionTree<T>, groupedBranchLabel: Map<RandomVariable, List<Pair<RVTest, Boolean>>>) : DecisionTree<Pair<RVAssignmentDisj, T>>{
        return when(node){
            is DTLeaf -> {
                val disjAssgn = groupedTestsToDisjAssignment(groupedBranchLabel)
                DTLeaf(Pair(disjAssgn, node.value))
            }
            is DTDecision -> {
                val rv = node.rvTest.first
                val newPassList = (groupedBranchLabel[rv] ?: emptyList()) + Pair(node.rvTest, true)
                val newFailList = (groupedBranchLabel[rv] ?: emptyList()) + Pair(node.rvTest, false)

                DTDecision(node.rvTest,
                    withBranchRec(node.passBranch, groupedBranchLabel + Pair(rv, newPassList)),
                    withBranchRec(node.failBranch, groupedBranchLabel + Pair(rv, newFailList))
                )
            }
        }
    }
    return withBranchRec(dt, emptyMap())
}

sealed class AugDT<out A, T>{
    abstract val augInfo: A
}
data class AugLeaf<out A, T>(override val augInfo : A, val leaf : DTLeaf<T>) : AugDT<A, T>()
data class AugDecision<out A, T>(override val augInfo : A, val rvTest : RVTest, val passBranch: AugDT<A, T>, val failBranch: AugDT<A, T>) : AugDT<A, T>()

fun <T> disjAssignmentAug(dt : DecisionTree<T>) : AugDT<RVAssignmentDisj, T>{
    fun augRec(node : DecisionTree<T>, groupedBranchLabel : Map<RandomVariable, List<Pair<RVTest, Boolean>>>) : AugDT<RVAssignmentDisj, T>{
        val rvDisj = groupedTestsToDisjAssignment(groupedBranchLabel)

        return when(node){
            is DTLeaf -> AugLeaf(rvDisj, node)
            is DTDecision -> {
                val rv = node.rvTest.first
                val newPassList = (groupedBranchLabel[rv] ?: emptyList()) + Pair(node.rvTest, true)
                val newFailList = (groupedBranchLabel[rv] ?: emptyList()) + Pair(node.rvTest, false)

                val augPass = augRec(node.passBranch, groupedBranchLabel + Pair(rv, newPassList))
                val augFail = augRec(node.failBranch, groupedBranchLabel + Pair(rv, newFailList))
                return AugDecision(rvDisj, node.rvTest, augPass, augFail)
            }
        }
    }

    return augRec(dt, emptyMap())
}

fun groupedTestsToDisjAssignment(groupedTests : Map<RandomVariable, List<Pair<RVTest, Boolean>>>) : RVAssignmentDisj =
    groupedTests.mapValues { (rv, rvTests) ->
        val passedTest : Pair<RVTest, Boolean>? = rvTests.find { (_, pass) -> pass }
        if(passedTest != null)
            setOf(passedTest.first.second)
        else{
            val failedTestVals = rvTests.map { it.first.second }
            (rv.domain.indices - failedTestVals).toSet()
        }
    }

fun probability(branchAssgnDisj: RVAssignmentDisj, factorMap : NaiveFactors) : Double{
    val prob = branchAssgnDisj
        .map { (rv, assgnVals) -> if(factorMap[rv] != null) assgnVals.sumByDouble { factorMap[rv]!!.values[it] } else 1.0 }
        .productByDouble { it }
    if(prob > 0 && branchAssgnDisj.keys.size > factorMap.keys.size){
        throw IllegalArgumentException("PLeaf should have information on this value node!")
    }
    return prob
}

fun <T> matchLeaf(dTree : DecisionTree<T>, context: RVAssignment) : DTLeaf<T> =
    when(dTree){
        is DTLeaf -> dTree
        is DTDecision -> {
            if(dTree.rvTest.first !in context){
                throw IllegalArgumentException("Tree trying to match against RV not present in context: ${dTree.rvTest}")
            }
            if(context[dTree.rvTest.first]!! == dTree.rvTest.second){
                matchLeaf(dTree.passBranch, context)
            }
            else{
                matchLeaf(dTree.failBranch, context)
            }
        }
    }


fun unifStartIDTransJoint(vocab : Collection<RandomVariable>) =
    vocab.associate { Pair(it, unifStartIDTransJoint(it)) }

fun unifStartIDTransJoint(rv : RandomVariable) : DecisionTree<Factor> {
    fun unifStartRec(remainingDomain : List<Int>) : DecisionTree<Factor> {
        val condFac = detFactor(rv, remainingDomain[0])
        val jointFac = Factor(condFac.scope, condFac.values.map { it / rv.domainSize })
        val detLeaf = DTLeaf(jointFac)

        return when(remainingDomain.size){
            1 -> detLeaf
            else -> DTDecision(Pair(rv, remainingDomain[0]), detLeaf, unifStartRec(remainingDomain.drop(1)))
        }
    }

    return unifStartRec(rv.domain.indices.toList())
}

/*
fun idTransJoint(rv : RandomVariable) : DecisionTree<Factor> {
    fun transRec(remainingDomain : List<Int>) : DecisionTree<Factor> =
        when(remainingDomain.size){
            1 -> DTLeaf(detFactor(rv, remainingDomain[0]))
            else -> DTDecision(Pair(rv, remainingDomain[0]),
                DTLeaf(detFactor(rv, remainingDomain[0])),
                transRec(remainingDomain.drop(1)))
        }

    return transRec(rv.domain.indices.toList())
}
*/

fun identityTransition(rv : RandomVariable, noise : Double = 0.0) : DecisionTree<Factor> {
    fun idTransRec(remainingDomain : List<Int>) : DecisionTree<Factor> =
        when(remainingDomain.size){
            1 -> DTLeaf(detFactor(rv, remainingDomain[0], noise))
            else -> DTDecision(Pair(rv, remainingDomain[0]),
                DTLeaf(detFactor(rv, remainingDomain[0], noise)),
                idTransRec(remainingDomain.drop(1)))
        }

    return idTransRec(rv.domain.indices.toList())
}

fun detFactor(rv : RandomVariable, value : Int, noise: Double = 0.0) : Factor {
    if(noise < 0 || noise > 1){
        throw IllegalArgumentException("Noise val must be between 0 and 1")
    }
    return Factor(listOf(rv), rv.domain.indices.map { if(it == value) (1 - noise) + (noise * (1.0 / rv.domainSize))  else noise * (1.0 / rv.domainSize)})
}

/*
fun pRegressAlternative(valTree : DecisionTree<Double>, dbn : DynamicBayesNet) : DecisionTree<NaiveFactors> {
    val relevantVars = vocabInDT(valTree).toSet()
    val relevantProbTrees = dbn.filterKeys { it in relevantVars }
    val toMerge = relevantProbTrees.map { (rv, cpt) -> fMap(cpt, { mapOf(rv to it)})}
    val pTree = merge(toMerge, { map1, map2 -> map1 + map2}, ::doubleEquality)
    return pTree
}
*/

fun <T> pRegress(valTree : DecisionTree<T>, dbn: DynamicBayesNet, vBranchLabel : Map<RVTest, Boolean> = emptyMap()) : DecisionTree<NaiveFactors>{
    when(valTree){
        is DTLeaf -> return DTLeaf(emptyMap())
        is DTDecision -> {
            val rootVar = valTree.rvTest.first
            val dbnCPDTree = fMap(dbn[rootVar]!!, { mapOf(it.scope.first() to it) })

            if(equivStructure(valTree.passBranch, valTree.failBranch)){
                val toAppend = pRegress(valTree.passBranch, dbn, vBranchLabel + Pair(valTree.rvTest, true))
                return append(dbnCPDTree, toAppend, {map1, map2 -> map1 + map2}, ::doubleEquality)
            }

            // Step 3: For values of the root variable that occur with some positive probability
            val passPTree = pRegress(valTree.passBranch, dbn, vBranchLabel + Pair(valTree.rvTest, true))
            val failPTree = pRegress(valTree.failBranch, dbn, vBranchLabel + Pair(valTree.rvTest, false))
            var mergedPTree : DecisionTree<NaiveFactors>? = null //merge(listOf(passPTree, failPTree), { map1, map2 -> map1 + map2}, ::doubleEquality)

            return leavesWithHistory(dbnCPDTree)
                .fold(dbnCPDTree, { currentTree, (dbnBranchLabel,leaf) ->
                    // Given the current leaf probabilities, which branches are even possible?
                    val leafCPD = leaf.value[rootVar]!!

                    val passRelevant = leafCPD.values[valTree.rvTest.second] > 0.0

                    val failsSoFar = vBranchLabel.keys.filter { it.first == rootVar }.map { it.second }.toSet() + valTree.rvTest.second
                    val failRelevant = leafCPD.values
                        .withIndex()
                        .any { (i, p) -> (i !in failsSoFar) && p > 0.0 }

                    when{
                        passRelevant && failRelevant -> {
                            if(mergedPTree == null){
                                mergedPTree = merge(listOf(passPTree, failPTree), { map1, map2 -> map1 + map2}, ::doubleEquality)
                            }
                            appendToLeaf(currentTree, dbnBranchLabel, mergedPTree!!, {map1, map2 -> map1 + map2}, ::doubleEquality)
                        }
                        passRelevant -> appendToLeaf(currentTree, dbnBranchLabel, passPTree, {map1, map2 -> map1 + map2}, ::doubleEquality)
                        failRelevant -> appendToLeaf(currentTree, dbnBranchLabel, failPTree, {map1, map2 -> map1 + map2}, ::doubleEquality)
                        else -> currentTree
                    }
                })

        }
    }
}

fun <T> equivStructure(dt1 : DecisionTree<T>, dt2 : DecisionTree<T>) : Boolean {
    if(dt1 is DTLeaf && dt2 is DTLeaf){
        return true
    }

    if(dt1 is DTDecision && dt2 is DTDecision && dt1.rvTest == dt2.rvTest){
        return equivStructure(dt1.passBranch, dt2.passBranch) && equivStructure(dt1.failBranch, dt2.failBranch)
    }

    return false
}

fun <T> simplify(dt : DecisionTree<T>, equalityTest : (T, T) -> Boolean, branchHistory : Map<RVTest, Boolean> = emptyMap()) : DecisionTree<T>{
    when(dt){
        is DTLeaf -> return dt
        is DTDecision -> {
            if (dt.rvTest in branchHistory){
                val keepBranch = if(branchHistory[dt.rvTest]!!) dt.passBranch else dt.failBranch
                return simplify(keepBranch, equalityTest, branchHistory)
            }

            // If a previous test with this RV passed, then this different one will definitely fail
            val dtRV = dt.rvTest.first
            val prevRvPass = branchHistory.entries.find { it.key.first == dtRV && it.value }
            if(prevRvPass != null){
                return simplify(dt.failBranch, equalityTest, branchHistory)
            }

            // If you've tested failure on every other value of rv, you can (probably) safely infer this one passes
            val testedRVDomain = branchHistory.keys.filter { it.first == dtRV }

            if(testedRVDomain.size == dt.rvTest.first.domainSize - 1){
                return simplify(dt.passBranch, equalityTest, branchHistory)
            }

            val simplifiedPass = simplify(dt.passBranch, equalityTest, branchHistory + Pair(dt.rvTest, true))
            val simplifiedFail = simplify(dt.failBranch, equalityTest, branchHistory + Pair(dt.rvTest, false))

            if(checkEquality(simplifiedPass, simplifiedFail, equalityTest)){
                return simplifiedPass
            }

            return DTDecision(dt.rvTest, simplifiedPass, simplifiedFail)
        }
    }
}

fun <T> oneStepSimplify(dt : DecisionTree<T>, equalityTest : (T, T) -> Boolean) : DecisionTree<T> =
    when(dt){
        is DTLeaf -> dt
        is DTDecision ->
            if(checkEquality(dt.passBranch, dt.failBranch, equalityTest)) dt.passBranch
            else dt
    }

fun toRanged(dt : DecisionTree<Double>) : DecisionTree<Range> =
    fMap(dt, { Range(it, it)})

fun fromRanged(dt : DecisionTree<Range>) : DecisionTree<Double> =
    fMap(dt, {(it.upper + it.lower) / 2.0})

@JvmName("Prune Single")
fun prune(dt : DecisionTree<Double>, threshold : Double) =
    prune(fMap(dt, { Range(it, it) }), threshold)

fun prune(dt : DecisionTree<Range>, threshold : Double) : DecisionTree<Range> =
    when(dt){
        is DTLeaf -> dt
        is DTDecision -> {
            val prunedPass = prune(dt.passBranch, threshold)
            val prunedFail = prune(dt.failBranch, threshold)

            if (prunedPass is DTLeaf && prunedFail is DTLeaf){
                val passRange = prunedPass.value
                val failRange = prunedFail.value
                val maxVal : Double = maxOf(passRange.upper, failRange.upper)
                val minVal : Double = minOf(passRange.lower, failRange.lower)
                if(maxVal - minVal < threshold){
                    DTLeaf(Range(minVal, maxVal))
                }
                else {
                    DTDecision(dt.rvTest, prunedPass, prunedFail)
                }
            }
            else {
                DTDecision(dt.rvTest, prunedPass, prunedFail)
            }
        }
    }


fun transposeTree(dt : DTDecision<Range>, replacementTest : RVTest) : DecisionTree<Range>{
    val oldTest = dt.rvTest
    if(oldTest == replacementTest) return dt

    val passBranch = if(dt.passBranch is DTDecision) transposeTree(dt.passBranch, replacementTest) else dt.passBranch
    val failBranch = if(dt.failBranch is DTDecision) transposeTree(dt.failBranch, replacementTest) else dt.failBranch

    if(passBranch is DTDecision && failBranch is DTDecision){
        return dt.copy(rvTest = replacementTest,
                passBranch = DTDecision(oldTest, passBranch.passBranch, failBranch.passBranch),
                failBranch = DTDecision(oldTest, passBranch.failBranch, failBranch.failBranch)
        )
    }
    if(passBranch is DTLeaf && failBranch is DTLeaf) {
        return DTDecision(replacementTest,
            DTDecision(oldTest, passBranch, failBranch),
            DTDecision(oldTest, passBranch, failBranch)
        )
    }

    if(failBranch is DTDecision && passBranch is DTLeaf){
        return DTDecision(replacementTest,
            DTDecision(oldTest, passBranch, failBranch.passBranch),
            DTDecision(oldTest, passBranch, failBranch.failBranch)
        )
    }
    if(passBranch is DTDecision && failBranch is DTLeaf){
        return DTDecision(replacementTest,
            DTDecision(oldTest, passBranch.passBranch, failBranch),
            DTDecision(oldTest, passBranch.failBranch, failBranch)
        )
    }

    throw IllegalStateException("All type cases should be covered, how did you get here?")
}

/*
@JvmName("transposeAction")
fun transposeTree(dt : DTDecision<Action>, replacementTest : RVTest) : DecisionTree<Action>{
    val oldTest = dt.rvTest
    if(oldTest == replacementTest) return dt

    val passBranch = if(dt.passBranch is DTDecision) transposeTree(dt.passBranch, replacementTest) else dt.passBranch
    val failBranch = if(dt.failBranch is DTDecision) transposeTree(dt.failBranch, replacementTest) else dt.failBranch

    if(passBranch is DTDecision && failBranch is DTDecision){
        return dt.copy(rvTest = replacementTest,
            passBranch = DTDecision(oldTest, passBranch.passBranch, failBranch.passBranch),
            failBranch = DTDecision(oldTest, passBranch.failBranch, failBranch.failBranch)
        )
    }
    WRONG!!!
    if(passBranch is DTLeaf && failBranch is DTLeaf) {
        return DTDecision(replacementTest, passBranch, failBranch)
    }

    if(failBranch is DTDecision && passBranch is DTLeaf){
        return DTDecision(replacementTest,
            DTDecision(oldTest, passBranch, failBranch.passBranch),
            DTDecision(oldTest, passBranch, failBranch.failBranch)
        )
    }
    if(passBranch is DTDecision && failBranch is DTLeaf){
        return DTDecision(replacementTest,
            DTDecision(oldTest, passBranch.passBranch, failBranch),
            DTDecision(oldTest, passBranch.failBranch, failBranch)
        )
    }

    throw IllegalStateException("All type cases should be covered, how did you get here?")
}
*/

@JvmName("ReOrderDTSingle")
fun reOrderDT(dt : DecisionTree<Double>) = reOrderDT(fMap(dt, { Range(it, it)}))

fun reOrderDT(dt : DecisionTree<Range>) : DecisionTree<Range> {
    fun reOrderRec(node : DecisionTree<Range>) : DecisionTree<Range> {
        when(node){
            is DTLeaf -> return node
            is DTDecision -> {
                val dtTests = getAllTests(node)
                val varRanges = rangesForVarSplits(dtTests, node)
                val varSpans = varRanges.mapValues { (_, ranges) -> ranges.map { it.upper - it.lower } }
                val varEntropies = varSpans.mapValues { (_, spans) -> spanEntropy(spans) }
                val (bestTest, _) = varEntropies.maxBy { it.value }!!

                val transposedTree = transposeTree(node, bestTest)
                return when(transposedTree){
                    is DTLeaf -> transposedTree
                    is DTDecision -> transposedTree.copy(passBranch = reOrderRec(transposedTree.passBranch), failBranch = reOrderRec(transposedTree.failBranch))
                }
            }
        }
    }
    return simplify(reOrderRec(dt), ::doubleEquality)
}

fun spanEntropy(vals : Collection<Double>) : Double {
    if(vals.isEmpty()){
        throw IllegalArgumentException("Must contain at least one value!")
    }
    return vals.map { -Math.log(it) }.average()
}

fun intEntropy(vals : Collection<Int>) = spanEntropy(vals.map(Int::toDouble))

fun <T> getAllTests(dt : DecisionTree<T>) : Set<RVTest>{
    when(dt){
        is DTLeaf -> return emptySet()
        is DTDecision -> return getAllTests(dt.passBranch) + getAllTests(dt.failBranch) + dt.rvTest
    }
}

fun rangesForVarSplits(tests : Set<RVTest>, dt : DecisionTree<Range>) : Map<RVTest, List<Range>>{
    val rangeSplits = tests
        .associate { rvTest -> Pair(rvTest, listOf(false, true)
            .map { Range(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY) }
            .toMutableList())
        }
        .toMutableMap()

    fun splitsForVarsRec(node : DecisionTree<Range>, branchLabel : Map<RVTest, Boolean>){
        when(node){
            is DTLeaf -> {
                for(rvTest in tests){
                    val rvSplit = rangeSplits[rvTest]!!
                    if(branchLabel.containsKey(rvTest)){
                        val testLabel = if(branchLabel[rvTest]!!) 1 else 0
                        val relevantRange = rvSplit[testLabel]
                        rvSplit[testLabel] = Range(Math.min(relevantRange.lower, node.value.lower), Math.max(relevantRange.upper, node.value.upper))
                    }
                    else{
                        rangeSplits[rvTest] = rvSplit.map { Range(Math.min(it.lower, node.value.lower), Math.max(it.upper, node.value.upper)) }.toMutableList()
                    }
                }
            }
            is DTDecision -> {
                splitsForVarsRec(node.passBranch, branchLabel + Pair(node.rvTest, true))
                splitsForVarsRec(node.failBranch, branchLabel + Pair(node.rvTest, false))
            }
        }
    }

    splitsForVarsRec(dt, emptyMap())
    return rangeSplits
}

fun <T> parMerge(dts: Collection<DecisionTree<T>>, commutativeMerge : (T, T) -> T, equalityTest : (T, T) -> Boolean) : DecisionTree<T> =
    dts.parallelStream().reduce{acc, dt -> append(acc, dt, commutativeMerge, equalityTest)}.get()

fun <T> merge(dts : Collection<DecisionTree<T>>, mergeFunc: (T, T) -> T, equalityTest: (T, T) -> Boolean) =
    dts.reduce{acc, dt -> append(acc, dt, mergeFunc, equalityTest)}

fun <T> orderedMerge(testOrder: List<RVTest>, orderedDTs : List<DecisionTree<T>>, mergeFunc: (T, T) -> T)  : DecisionTree<T>{
    // Reminder : Do we need to simplify?
    if (orderedDTs.all { it is DTLeaf }){
        return (orderedDTs as List<DTLeaf<T>>).reduce { acc : DTLeaf<T>, dtLeaf : DTLeaf<T> -> DTLeaf(mergeFunc(dtLeaf.value, acc.value)) }
    }
    if(testOrder.isEmpty()){
        throw IllegalStateException("Ordering is used up, but there are still untraversed branches!")
    }

    val firstRVTest = testOrder.find { test -> orderedDTs.any { it is DTDecision && it.rvTest ==  test} }!!
    val passDTs = orderedDTs.map { dt -> if(dt is DTDecision && dt.rvTest == firstRVTest) dt.passBranch else dt }
    val failDTs = orderedDTs.map { dt -> if(dt is DTDecision && dt.rvTest == firstRVTest) dt.failBranch else dt }

    val remainingOrdering = testOrder.dropWhile { it != firstRVTest }
    return DTDecision(firstRVTest, orderedMerge(remainingOrdering, passDTs, mergeFunc), orderedMerge(remainingOrdering, failDTs, mergeFunc))
}

fun setOrder(dt : DecisionTree<Range>, order : List<RVTest>) : DecisionTree<Range> {
    fun setOrderRec(node : DecisionTree<Range>, remainingOrder : List<RVTest>) : DecisionTree<Range>{
        when(node){
            is DTLeaf -> return node
            is DTDecision -> {
                val dtTests = getAllTests(node)
                val relevantOrder = remainingOrder.filter { it in dtTests }
                if(relevantOrder.isEmpty()){
                    throw IllegalStateException("There are tests in this DT which are not mentioned in the ordering")
                }
                val transposedDT = transposeTree(node, relevantOrder.first())
                return when(transposedDT){
                    is DTLeaf -> transposedDT
                    is DTDecision -> transposedDT.copy(passBranch = setOrderRec(transposedDT.passBranch, relevantOrder.drop(1)), failBranch = setOrderRec(transposedDT.failBranch, relevantOrder.drop(1)))
                }
            }
        }
    }

    return simplify(setOrderRec(dt, order), ::doubleEquality)
}

fun <T, S> append(dt1 : DecisionTree<T>, dt2 : DecisionTree<T>, mergeFunc: (T, T) -> S, equalityTest: (S, S) -> Boolean) : DecisionTree<S>{
    fun appendRec(dt: DecisionTree<T>, branchLabel: Map<RVTest, Boolean>) : DecisionTree<S> {
        return when(dt){
            is DTLeaf<T> -> {
                val extendedLeaf = fMap(dt2, {mergeFunc(dt.value, it)})
                simplify(extendedLeaf, equalityTest, branchLabel)
            }
            is DTDecision<T> -> {
                /*
                val (newPass, newFail) = listOf(Pair(dt.passBranch, true), Pair(dt.failBranch, false))
                    .parallelStream()
                    .map { (b, passVal) -> appendRec(b, branchLabel + Pair(dt.rvTest, passVal)) }
                    .toList()
                */
                val extendedNode = DTDecision(dt.rvTest, appendRec(dt.passBranch, branchLabel + Pair(dt.rvTest, true)), appendRec(dt.failBranch, branchLabel + Pair(dt.rvTest, false)))
                oneStepSimplify(extendedNode, equalityTest)
            }
        }
    }

    return appendRec(dt1, emptyMap())
}

fun <T> appendToLeaf(dt : DecisionTree<T>, branchLabel: Map<RVTest, Boolean>, newNode: DecisionTree<T>, mergeFunc: (T, T) -> T, equalityTest: (T, T) -> Boolean) : DecisionTree<T> {
    fun appendRec(node : DecisionTree<T>) : DecisionTree<T> {
        return when(node){
            is DTLeaf<T> -> {
                val extendedLeaf = fMap(newNode, {mergeFunc(node.value, it)})
                simplify(extendedLeaf, equalityTest, branchLabel)
            }
            is DTDecision<T> -> {
                val extendedNode = if(branchLabel[node.rvTest]!!)
                    DTDecision(node.rvTest, appendRec(node.passBranch), node.failBranch)
                else
                    DTDecision(node.rvTest, node.passBranch, appendRec(node.failBranch))
                oneStepSimplify(extendedNode, equalityTest)
            }
        }
    }

    return appendRec(dt)
}

fun <T, S> fMap(dt : DecisionTree<T>, func : (T) -> S) : DecisionTree<S> =
    when(dt){
        is DTLeaf<T> -> DTLeaf(func(dt.value))
        is DTDecision<T> -> {
            DTDecision(dt.rvTest, fMap(dt.passBranch, func), fMap(dt.failBranch, func))
        }
    }

fun <T> checkEquality(dt1 : DecisionTree<T>, dt2 : DecisionTree<T>, eqFunc : (T, T) -> Boolean) : Boolean =
    when{
        dt1 is DTLeaf && dt2 is DTLeaf -> eqFunc(dt1.value, dt2.value)
        dt1 is DTDecision && dt2 is DTDecision && dt1.rvTest == dt2.rvTest -> {
            checkEquality(dt1.passBranch, dt2.passBranch, eqFunc) && checkEquality(dt1.failBranch, dt2.failBranch, eqFunc)
        }
        else -> false
    }

fun doubleEquality(a: Double, b: Double) : Boolean{
    val epsilon = 10E-7
    val absA = Math.abs(a)
    val absB = Math.abs(b)
    val diff = Math.abs(a - b)

    if(a == b){ // Shortcut to handle infinities
        return true
    }

    if(a == 0.0 || b == 0.0 || diff < Double.MIN_VALUE){
        return diff < (epsilon * Double.MIN_VALUE)
    }

    return diff / Math.min(absA + absB, Double.MAX_VALUE) < epsilon
}

fun doubleEquality(r1 : Range, r2: Range) =
    doubleEquality(r1.lower, r2.lower) && doubleEquality(r1.upper, r2.upper)

fun doubleEquality(f1 : Factor, f2: Factor) =
    f1.scope == f2.scope &&
        f1.values.size == f2.values.size &&
        f1.values.zip(f2.values)
            .all { (d1, d2) -> doubleEquality(d1, d2) }

fun doubleEquality(f1s : NaiveFactors, f2s : NaiveFactors) =
    f1s.keys == f2s.keys && f1s.keys.all { k -> doubleEquality(f1s[k]!!, f2s[k]!!) }

fun <T> numLeaves(dt : DecisionTree<T>) : Int{
    when(dt){
        is DTLeaf -> return 1
        is DTDecision -> return numLeaves(dt.passBranch) + numLeaves(dt.failBranch)
    }
}

fun <T> numNodes(dt : DecisionTree<T>) : Int {
    when(dt){
        is DTLeaf -> return 1
        is DTDecision -> return 1 + numLeaves(dt.passBranch) + numLeaves(dt.failBranch)
    }
}
