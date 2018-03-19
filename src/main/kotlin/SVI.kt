import Utils.allEqual
import Utils.concat
import Utils.productBy
import Utils.productByDouble

sealed class DecisionTree<T>
data class DTLeaf<T>(val value: T) : DecisionTree<T>()
data class DTDecision<T>(val rv: RandomVariable, val branches : List<DecisionTree<T>>) : DecisionTree<T>()
typealias NaiveFactors = Map<RandomVariable, Factor>
typealias Action = String
typealias QTree = DecisionTree<Double>
typealias VTree = DecisionTree<Double>
typealias PolicyTree = DecisionTree<Action>

fun <T> vocabInDT(dt : DecisionTree<T>) : PSet =
    when(dt){
        is DTLeaf -> emptySet()
        is DTDecision -> dt.branches.flatMap { vocabInDT(it) }.toSet() + dt.rv
    }

fun <T> leavesWithHistory(decisionTree: DecisionTree<T>, currentHistory : RVAssignment = emptyMap()) : List<Pair<RVAssignment, DTLeaf<T>>>{
    when(decisionTree){
        is DTLeaf<T> -> return listOf(Pair(currentHistory, decisionTree))
        is DTDecision<T> -> return decisionTree.branches
            .mapIndexed { x, node -> leavesWithHistory(node, currentHistory + Pair(decisionTree.rv, x)) }
            .flatten()
    }
}


fun <T, S> foldTree (dTree : DecisionTree<T>, initial : S, accFunc : (S, T) -> S) : S =
    when(dTree){
        is DTLeaf -> accFunc(initial, dTree.value)
        is DTDecision -> dTree.branches.fold(initial, {maxSoFar, dt -> foldTree(dt, maxSoFar, accFunc)})
    }

typealias DynamicBayesNet = Map<RandomVariable, DecisionTree<Factor>>

// Note, this assumes that all the random variables in your problem are binary, so conversion from binary test to assignments is trivial.
// To handle many values RVs, we will have to do something a bit more complicated
fun convertFromITI(binaryRTree : RewardTree) : DecisionTree<Reward> =
    when(binaryRTree){
        is RLeaf -> {
            val averageReward = if(binaryRTree.trials.isEmpty()){
                0.0
            }
            else{
                binaryRTree.trials.map { it.reward }.average()
            }
            DTLeaf(averageReward)

        } // This would be quicker if these stats had been pre-computed at leaf
        is RDecision -> {
            val orderedBranches = if(binaryRTree.bestTest.second == 0){
                listOf(binaryRTree.passBranch, binaryRTree.failBranch)
            }
            else{
                listOf(binaryRTree.failBranch, binaryRTree.passBranch)
            }
            DTDecision(binaryRTree.bestTest.first, orderedBranches.map(::convertFromITI))
        }
    }

fun convertToCPT(binaryPTree: ProbTree, priorParams: Map<RVAssignment, Factor>, pseudoCountStrength: Double, assignmentSoFar: RVAssignment = emptyMap()) : DecisionTree<Factor> =
    when(binaryPTree){
        is PLeaf -> {
            val matchingFactors = matchingAssignments(assignmentSoFar, priorParams)
            // Add matching factors together using factor function used in create p info
            val relevantPrior = add(matchingFactors)
            val bestParams = maxPosteriorParams(binaryPTree.rv, binaryPTree.counts, relevantPrior, pseudoCountStrength)
            DTLeaf(bestParams)
        }
        is PDecision -> {
            if(binaryPTree.rv.domainSize != 2){
                throw NotImplementedError("Can't yet handle non-binary variables")
            }
            val orderedBranches = if(binaryPTree.currentTest.second == 0){
                listOf(binaryPTree.passBranch, binaryPTree.failBranch)
            }
            else{
                listOf(binaryPTree.failBranch, binaryPTree.passBranch)
            }
            DTDecision(binaryPTree.rv, orderedBranches.mapIndexed { i, branch -> convertToCPT(branch, priorParams, pseudoCountStrength, assignmentSoFar + Pair(binaryPTree.currentTest.first, i))} )
        }
    }

// Take a decision tree for joint probabilities, find all leaves that match assignment, and combine them
// For leaves that make no assignment to some vars in parent assignment, assume counts would be uniformly distributed over missing context
fun jointQuery(parentAssgn : RVAssignment, jointDT: DecisionTree<Factor>) : Factor{
    fun jointQueryRec(dt : DecisionTree<Factor>, assignmentSoFar: RVAssignment) : Factor =
        when(dt){
            is DTLeaf -> {
                val unmatchedContext = (parentAssgn.keys - assignmentSoFar.keys).toList()
                val divAmount = unmatchedContext.productByDouble { it.domainSize.toDouble() }
                val scaledVals = dt.value.values.map { it / divAmount }
                Factor(dt.value.scope, scaledVals)
            }
            is DTDecision -> {
                if(parentAssgn.containsKey(dt.rv)){
                    val rvVal = parentAssgn[dt.rv]!!
                    jointQueryRec(dt.branches[rvVal], assignmentSoFar + Pair(dt.rv, rvVal))
                }
                else{
                    add(dt.branches.map { jointQueryRec(it, assignmentSoFar) })
                }
            }
        }
    return jointQueryRec(jointDT, emptyMap())
}

fun convertToJointProbTree(binaryPTree : ProbTree, priorJointDT: DecisionTree<Factor>, psuedoCountStrength : Double) : DecisionTree<Factor> {
    val totalTrials = when(binaryPTree){
        is PLeaf -> binaryPTree.trials.size
        is PDecision -> {
            val testStat = binaryPTree.testCandidates[binaryPTree.currentTest]!!
            testStat.passTrialCounts.values.sum() + testStat.failTrialCounts.values.sum()
        }
    }

    fun convertToJointProbTreeRec(pt : ProbTree, parentAssgn: RVAssignment) :DecisionTree<Factor> =
        when(pt){
            is PLeaf -> {
                val jointPriorParam = jointQuery(parentAssgn, priorJointDT)
                val realCounts = pt.rv.domain.map { (pt.counts[it] ?: 0)}
                val scaledCombined = pt.rv.domain.map { (realCounts[it] + (jointPriorParam.values[it] * psuedoCountStrength)) / (totalTrials + psuedoCountStrength)}
                DTLeaf(Factor(listOf(pt.rv), scaledCombined))
            }
            is PDecision -> {
                if(pt.rv.domainSize != 2){
                    throw NotImplementedError("Can't yet handle non-binary variables")
                }
                val orderedBranches = if(pt.currentTest.second == 0){
                    listOf(pt.passBranch, pt.failBranch)
                }
                else{
                    listOf(pt.failBranch, pt.passBranch)
                }
                DTDecision(pt.rv, orderedBranches.mapIndexed { i, branch -> convertToJointProbTreeRec(branch, parentAssgn + Pair(pt.rv, i)) })
            }
        }
    return convertToJointProbTreeRec(binaryPTree, emptyMap())
}

fun <T> matchingAssignments(partialAssignment : RVAssignment, fullAssignments : Map<RVAssignment, T>) : List<T>{
    if(fullAssignments.isEmpty()){
        return emptyList()
    }
    val fullAssgnVars = fullAssignments.keys.first().keys
    val partialVars = partialAssignment.keys
    if(!fullAssgnVars.containsAll(partialVars)){
        throw IllegalArgumentException("Partial assignment keys must be subset of full assignment keys")
    }
    val completionVars = fullAssgnVars - partialVars
    return allAssignments(completionVars.toList())
        .map { cAssgn -> fullAssignments[partialAssignment + cAssgn]!! }
}

fun partialMatch(partialAssignment: RVAssignment, fullAssignment : RVAssignment) =
    partialAssignment == fullAssignment.filterKeys { it in partialAssignment.keys }

fun maxPosteriorParams(rv : RandomVariable, counts : Map<Int, Int>, priorParams : Factor, pseudoCountStrength : Double) : Factor{
    val totalCounts = rv.domain.sumBy { counts[it] ?: 0}
    val totalPseudoCounts = rv.domain.sumByDouble { priorParams.values[it] * pseudoCountStrength }
    val bestParams = rv.domain.map { rvVal -> ((counts[rvVal] ?: 0) + priorParams.values[rvVal] * pseudoCountStrength) / (totalCounts + totalPseudoCounts) }
    return Factor(listOf(rv), bestParams)
}

fun  regressPolicy(valTree: DecisionTree<Double>, rewardTree : DecisionTree<Double>, policyTree : DecisionTree<Action>, actionDBNs : Map<Action, DynamicBayesNet>) : DecisionTree<Double>{
    val qTrees = actionDBNs.mapValues { regress(valTree, rewardTree, it.value) }
    val compPolicyVal = replaceLeaves(policyTree, qTrees)
    val policyVals = simplify(compPolicyVal, ::doubleEquality)
    return policyVals
}

fun structuredSuccessiveApproximation(rewardTree : DecisionTree<Double>, policyTree : DecisionTree<Action>, actionDBNs : Map<Action, DynamicBayesNet>) : DecisionTree<Double> {
    var valueTree = rewardTree
    do{
        val newValueTree = regressPolicy(valueTree, rewardTree, policyTree, actionDBNs)
        val diffTree = append(valueTree, newValueTree, {a, b -> Math.abs(a - b)}, ::doubleEquality)
        val biggestDiff = foldTree(diffTree, 0.0, ::maxOf)

        valueTree = newValueTree
    }
    while(biggestDiff < 0.001)
    return valueTree
}

fun structuredPolicyIteration(rewardTree : DecisionTree<Double>, actionDBNs : Map<Action, DynamicBayesNet>) : DecisionTree<Action>{
    var currentPolicy : DecisionTree<Action> = DTLeaf(actionDBNs.keys.first()) // Start by deterministically doing action 0 in every state
    do{
        val valueTree = structuredSuccessiveApproximation(rewardTree, currentPolicy, actionDBNs)
        val qVals = actionDBNs.mapValues { regress(valueTree, rewardTree, it.value) }
        val newPolicy = choosePolicy(qVals)

        val policyChanged = newPolicy != currentPolicy

        currentPolicy = newPolicy
    } while (policyChanged)
    return currentPolicy
}

fun structuredValueIteration(rewardTree : DecisionTree<Double>, actionDBNs: Map<Action, DynamicBayesNet>) : Pair<PolicyTree, VTree>{
    var valueTree = rewardTree
    do{
        val qTrees = actionDBNs.map { regress(valueTree, rewardTree, it.value) }
        val newValueTree = merge(qTrees, ::maxOf, ::doubleEquality)

        val diffTree = append(valueTree, newValueTree, {a, b -> Math.abs(a - b)}, ::doubleEquality)
        val biggestDiff = foldTree(diffTree, 0.0, ::maxOf)

        valueTree = newValueTree
    } while(biggestDiff < 0.001)

    val finalQTrees = actionDBNs.mapValues { regress(valueTree, rewardTree, it.value) }
    val finalPolicy = choosePolicy(finalQTrees)
    return Pair(finalPolicy, valueTree)
}

fun incrementalSVI(rewardTree : DecisionTree<Double>, valueTree : DecisionTree<Double>, actionDBNs : Map<Action, DynamicBayesNet>, pruningThresh : Double = 0.0) : Pair<VTree, Map<Action, QTree>>{
    val prunedValueTree = prune(valueTree, pruningThresh)
    val qTrees = actionDBNs.mapValues { regress (prunedValueTree, rewardTree, it.value) }
    val newValueTree = merge(qTrees.values, ::maxOf, ::doubleEquality)
    return Pair(newValueTree, qTrees)
}

fun applyExpertAdvice(rewardTree : DecisionTree<Double>, valueTree : DecisionTree<Double>, actionDBNs : Map<Action, DynamicBayesNet>, actionAdvice: Pair<RVAssignment, Action>) : Pair<VTree, Map<Action, QTree>> {
    val qTrees = actionDBNs.mapValues { regress (valueTree, rewardTree, it.value) }.toMutableMap()

    val bestVal : Double = qTrees
        .map{ (_, qTree) -> matchLeaf(qTree, actionAdvice.first) }
        .maxBy { leaf -> leaf.value}!!.value

    qTrees[actionAdvice.second] = replaceLeafValue(qTrees[actionAdvice.second]!!, actionAdvice.first, bestVal)

    val newValueTree = merge(qTrees.values, ::maxOf, ::doubleEquality)
    return Pair(newValueTree, qTrees)
}

fun choosePolicy(qTrees : Map<Action, DecisionTree<Double>>) : PolicyTree {
    val annotatedActionQs = qTrees.map {(a, qt) -> fMap(qt, {qVal -> Pair(a, qVal)})}
    val annotatedPolicy = merge(annotatedActionQs, ::maxBySecond, {a, b -> a == b})
    return fMap(annotatedPolicy, {(action, _) -> action})
}

fun <T, S : Comparable<S>> maxBySecond(p1 : Pair<T,S>, p2 : Pair<T, S>) =
    if(p1.second >= p2.second) p1 else p2

fun replaceLeaves(policyTree : DecisionTree<Action>, qTrees : Map<Action, DecisionTree<Double>>) : DecisionTree<Double>{
    return when(policyTree){
        is DTLeaf -> qTrees[policyTree.value]!!
        is DTDecision -> DTDecision(policyTree.rv, policyTree.branches.map { replaceLeaves(it, qTrees) })
    }
}

fun <T> replaceLeafValue(dt : DecisionTree<T>, context : RVAssignment, replacementValue : T) : DecisionTree<T> =
    when(dt){
        is DTLeaf -> { dt.copy(value = replacementValue)}
        is DTDecision -> {
            dt.copy(branches = dt.branches.mapIndexed { i, branch ->
                if(context[dt.rv]!! == i) replaceLeafValue(branch, context, replacementValue) else branch
            })
        }
    }

fun regress(valTree : DecisionTree<Double>, rewardTree : DecisionTree<Double>, dbn : DynamicBayesNet) :  QTree {
    val discountFactor = 0.9
    val pTree = pRegress(valTree, dbn)
    val fvTree = undiscountedFutureValue(valTree, pTree)
    val newTree = append(fvTree, rewardTree, {v, r -> discountFactor * v + r}, ::doubleEquality)
    return newTree
}


fun undiscountedFutureValue(valueTree : DecisionTree<Double>, pTree : DecisionTree<NaiveFactors>) : DecisionTree<Double>{
    val vTreeBranches = leavesWithHistory(valueTree)
    val undiscountedVals = fMap(pTree, { pLeaf -> vTreeBranches
        .sumByDouble { (branchHist, vLeaf) -> vLeaf.value * probability(branchHist, pLeaf) }
    })
    return undiscountedVals
}

fun probability(assignment : RVAssignment, factorMap : NaiveFactors) : Double{
    return assignment
        .filterKeys { rv -> rv in factorMap }
        .map{ (rv, i) -> factorMap[rv]!!.values[i] }.productByDouble { it }
}

fun <T> matchLeaf(dTree : DecisionTree<T>, context: RVAssignment) : DTLeaf<T> =
    when(dTree){
        is DTLeaf -> dTree
        is DTDecision -> matchLeaf(dTree.branches[context[dTree.rv]!!], context)
    }

fun <T> matchAllLeaves(dTree : DecisionTree<T>, partialContext : RVAssignment) : List<DTLeaf<T>> =
    when(dTree){
        is DTLeaf -> listOf(dTree)
        is DTDecision -> if(partialContext.containsKey(dTree.rv)){
            matchAllLeaves(dTree.branches[partialContext[dTree.rv]!!], partialContext)
        }
        else{
            dTree.branches.flatMap { matchAllLeaves(it, partialContext) }
        }
    }

fun uniStartIdentityTransition(rv : RandomVariable, noise : Double = 0.0) : DTDecision<Factor> =
    DTDecision(rv, rv.domain.map { prevValue ->
        val condFac = detFactor(rv, prevValue, noise)
        val jointFac =  Factor(condFac.scope, condFac.values.map { it / rv.domainSize })
        DTLeaf(jointFac)
    })

fun identityTransition(rv : RandomVariable, noise : Double = 0.0) : DTDecision<Factor> =
    DTDecision(rv, rv.domain.map { prevValue ->  DTLeaf(detFactor(rv, prevValue, noise))})

fun detFactor(rv : RandomVariable, value : Int, noise: Double = 0.0) : Factor {
    if(noise < 0 || noise > 1){
        throw IllegalArgumentException("Noise val must be between 0 and 1")
    }
    return Factor(listOf(rv), rv.domain.map { if(it == value) (1 - noise) + (noise * (1.0 / rv.domainSize))  else noise * (1.0 / rv.domainSize)})
}

fun pRegress(valTree : DecisionTree<Double>, dbn : DynamicBayesNet) : DecisionTree<NaiveFactors>{
    // A single leaf node value means you will always arrive at this value regardless of assignment
    if(valTree is DTLeaf<Double>)
        return DTLeaf(emptyMap())
    val rootDecision = valTree as DTDecision<Double>
    val rootVar = rootDecision.rv
    val dbnCPDTree = dbn[rootVar]!!

    // Step 3: For values of the root variable that occur with some positive probability
    val subTrees = rootVar.domain.map{ rootDecision.branches[it]}
    val subPTrees = subTrees.map { pRegress(it, dbn) }

    // Step 4: Loop over every leaf in cpd tree for root variable
    val initialPTree = fMap(dbnCPDTree, {mapOf(it.scope.first() to it)})
    return leavesWithHistory(initialPTree)
        .fold(initialPTree, { currentTree, (_, leafCPD) ->
            val relevantSubPTrees = subPTrees.filterIndexed { i, _ -> leafCPD.value[rootVar]!!.values[i] > 0.0}
            val mergedSubPTrees = merge(relevantSubPTrees, {map1, map2 -> map1 + map2}, ::doubleEquality)
            append(currentTree, leafCPD, mergedSubPTrees, {map1, map2 -> map1 + map2}, ::doubleEquality)
        })
}

fun <T> simplify(dt : DecisionTree<T>, equalityTest : (T, T) -> Boolean, branchHistory : RVAssignment = emptyMap()) : DecisionTree<T>{
    when(dt){
        is DTLeaf -> return dt
        is DTDecision -> {
            if (dt.rv in branchHistory.keys){
                return simplify(dt.branches[branchHistory[dt.rv]!!], equalityTest, branchHistory)
            }
            else{
                val simplifyFirstPass = DTDecision(dt.rv, dt.branches.mapIndexed{i, node -> simplify(node, equalityTest, branchHistory + Pair(dt.rv, i))})
                val headBranch = simplifyFirstPass.branches.first()
                val tailBranches = simplifyFirstPass.branches.subList(1, simplifyFirstPass.branches.size)

                if(tailBranches.all { checkEquality(headBranch, it, equalityTest) }){
                    return simplifyFirstPass.branches.first()
                }
                return simplifyFirstPass
            }
        }
    }
}

fun prune(dt : DecisionTree<Double>, threshold : Double) : DecisionTree<Double> {
    fun pruneRanged(rt : DecisionTree<Double>) : DecisionTree<Pair<Double, Double>> {
        when(rt){
            is DTLeaf -> return DTLeaf(Pair(rt.value, rt.value))
            is DTDecision -> {
                val prunedBranches = rt.branches.map(::pruneRanged)
                if (!prunedBranches.all { it is DTLeaf }){
                    return DTDecision(rt.rv, prunedBranches)
                }
                val childRanges = prunedBranches.map { (it as DTLeaf<Pair<Double, Double>>).value }
                val maxVal : Double = childRanges.map { it.second }.max()!!
                val minVal : Double = childRanges.map { it.first }.min()!!
                if(maxVal - minVal < threshold){
                    return DTLeaf(Pair(minVal, maxVal))
                }
                return DTDecision(rt.rv, prunedBranches)
            }
        }
    }

    return fMap(pruneRanged(dt), {(min, max) -> (min + max) / 2.0})
}

fun <T> merge(dts : Collection<DecisionTree<T>>, mergeFunc: (T, T) -> T, equalityTest: (T, T) -> Boolean) =
    dts.reduce{acc, dt -> append(acc, dt, mergeFunc, equalityTest)}

fun <T> append(dt1 : DecisionTree<T>, dt2 : DecisionTree<T>, mergeFunc: (T, T) -> T, equalityTest: (T, T) -> Boolean) : DecisionTree<T>{
    val complicatedDT = when(dt1){
        is DTLeaf<T> -> fMap(dt2, {mergeFunc(dt1.value, it)})
        is DTDecision<T> -> DTDecision(dt1.rv, dt1.branches.map { append(it, dt2, mergeFunc, equalityTest)})
    }
    return simplify(complicatedDT, equalityTest)
}

fun <T> append(dt : DecisionTree<T>, oldLeaf : DTLeaf<T>, newNode: DecisionTree<T>, mergeFunc: (T, T) -> T, equalityTest: (T, T) -> Boolean) : DecisionTree<T> {
    val complicatedDT = when(dt){
        oldLeaf -> fMap(newNode, {mergeFunc(oldLeaf.value, it)})
        is DTLeaf<T> -> dt
        is DTDecision<T> -> DTDecision(dt.rv, dt.branches.map { append(it, oldLeaf, newNode, mergeFunc, equalityTest) })
    }
    return simplify(complicatedDT, equalityTest)
}

fun <T, S> fMap(dt : DecisionTree<T>, func : (T) -> S) : DecisionTree<S> =
    when(dt){
        is DTLeaf<T> -> DTLeaf(func(dt.value))
        is DTDecision<T> -> DTDecision(dt.rv, dt.branches.map{fMap(it, func)})
    }

fun <T> checkEquality(dt1 : DecisionTree<T>, dt2 : DecisionTree<T>, eqFunc : (T, T) -> Boolean) : Boolean =
    when{
        dt1 is DTLeaf && dt2 is DTLeaf -> eqFunc(dt1.value, dt2.value)
        dt1 is DTDecision && dt2 is DTDecision && dt1.rv == dt2.rv && dt1.branches.size == dt2.branches.size -> {
            dt1.branches.zip(dt2.branches).all { (b1, b2) -> checkEquality(b1, b2, eqFunc) }
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

fun doubleEquality(f1 : Factor, f2: Factor) =
    f1.scope == f2.scope &&
        f1.values.size == f2.values.size &&
        f1.values.zip(f2.values)
            .all { (d1, d2) -> doubleEquality(d1, d2) }

fun doubleEquality(f1s : NaiveFactors, f2s : NaiveFactors) =
    f1s.keys == f2s.keys && f1s.keys.all { k -> doubleEquality(f1s[k]!!, f2s[k]!!) }
