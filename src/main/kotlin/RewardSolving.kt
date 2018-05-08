import Utils.*
import com.google.common.collect.BiMap
import com.google.common.collect.HashBiMap
import com.google.ortools.constraintsolver.DefaultPhaseParameters
import com.google.ortools.constraintsolver.Solver
import com.google.ortools.linearsolver.MPSolver
import com.google.ortools.linearsolver.MPVariable
import java.util.*

data class StateDisjunct(val rewardSymbol: Long, val states : List<Int>)
typealias Reward = Double
typealias RewardTable = RVTable<Reward>
typealias TimeStamp = Int

fun getReward(factors : List<RewardTable>, assignment : RVAssignment) =
    factors.sumByDouble { it.getValue(assignment) }

interface AssignmentReward {
    val assignment : RVAssignment
    val reward : Reward
}

data class Trial(override val assignment: RVAssignment, override val reward: Reward) : AssignmentReward

data class BetterActionAdvice(val action : RVAssignment, override val reward : Reward, val trialTime : TimeStamp) : AssignmentReward{
    override val assignment = action
}

data class RVTable<out V>(val scope: List<RandomVariable>, val values: List<V>) {
    init {
        if(scope.productBy { it.domainSize } != values.size)
            throw IllegalArgumentException("Must have value for every assignment to scope")
    }


    fun getValue(assignment : RVAssignment) =
        values[assignmentToIndex(assignment, scope)]

    fun getValue(rv : RandomVariable, rvVal : Int) : V {
        if(scope.size != 1 || rv != scope[0])
            throw IllegalArgumentException("Function only value for single variable scopes")

        return getValue(mapOf(rv to rvVal))
    }
}

fun totalValue(assignment : RVAssignment, rewardTables : List<RewardTable>) =
    rewardTables.sumByDouble { it.getValue(assignment) }

object FactoredRewardSolver {
    init {
        System.loadLibrary("jniortools")
    }

    internal val solver = MPSolver("Factored Reward Solver", MPSolver.OptimizationProblemType.CBC_MIXED_INTEGER_PROGRAMMING)
    fun solve(rewardDom: List<RandomVariable>, betterActions: List<BetterActionAdvice>, existsStates: List<Trial>, maxFactorSize: Int): List<RewardTable> {
        solver.clear()
        if (rewardDom.any { it.domainSize != 2 })
            throw IllegalArgumentException("Solver doesnt support non-binary variables yet")

        val potentialFactors = powerSet(rewardDom, 1, maxFactorSize)
            .map {
                RVTable(it, allAssignments(it)
                    .map { Pair(it, solver.makeNumVar(0.0, 100.0, assignmentAsString(it))) })
            }

        // Create objective function
        val objective = solver.objective()
        objective.setMinimization()

        val assignmentDecisionPairs = potentialFactors.flatMap { it.values }
        assignmentDecisionPairs.forEach { objective.setCoefficient(it.second, 1.0) }

        existsStates.forEach {
            if(it.assignment.keys.containsAll(rewardDom)) {
                singleConstraint(it, assignmentDecisionPairs, rewardDom, ConstraintType.EQ)
            }
            else{
                disjunctiveConstraint(it, assignmentDecisionPairs, rewardDom, ConstraintType.EQ)
            }
        }

        betterActions.forEach { disjunctiveConstraint(it, assignmentDecisionPairs, rewardDom, ConstraintType.GT) }

        // Actually compute solution
        val resultStatus = solver.solve()
        if (resultStatus != MPSolver.ResultStatus.OPTIMAL) {
            println(resultStatus)
            return emptyList()
        }

        val rewardTables = potentialFactors.map { (scope, vals) -> RewardTable(scope, vals.map { (_, mpVar) -> mpVar.solutionValue() }) }
        val finalResult = compressSingleEntries(reportSubsets(removeEmpty(rewardTables)))
        return finalResult
    }

    enum class ConstraintType { GT, EQ }

    private fun singleConstraint(state : AssignmentReward, assignmentDecisionPairs: List<Pair<RVAssignment, MPVariable>>, rewardDom : List<RandomVariable>, conType : ConstraintType){
        val epsilon = 10E-4
        val relevantDecisionVars = assignmentDecisionPairs.filter { (assgn, _) -> assgn.all { state.assignment[it.key] == it.value } }


        if (conType == ConstraintType.EQ) {
            val gtConstraint = solver.makeConstraint()
            gtConstraint.setLb(state.reward)
            relevantDecisionVars.forEach { gtConstraint.setCoefficient(it.second, 1.0) }

            // Less Than Constraint
            val ltConstraint = solver.makeConstraint()
            ltConstraint.setUb(state.reward)
            relevantDecisionVars.forEach { ltConstraint.setCoefficient(it.second, 1.0) }
        }
        else {
            val gtConstraint = solver.makeConstraint()
            gtConstraint.setLb(state.reward + epsilon)
            relevantDecisionVars.forEach { gtConstraint.setCoefficient(it.second, 1.0) }
        }
    }

    private fun disjunctiveConstraint(partialState: AssignmentReward, assignmentDecisionPairs: List<Pair<RVAssignment, MPVariable>>, rewardDom: List<RandomVariable>, conType: ConstraintType) {
        val bigM = 100.0
        val epsilon = 10E-4

        val unmentionedVars = rewardDom - partialState.assignment.keys
        val disjunctionVars = ArrayList<MPVariable>()

        // For each unmentioned rv, create a new binary rv
        for (completionAssgn in allAssignments(unmentionedVars)) {
            val disjunctionVar = solver.makeBoolVar("Disjunction ID: ${UUID.randomUUID()}")
            disjunctionVars.add(disjunctionVar)

            val fullAssignment = partialState.assignment + completionAssgn
            val relevantDecisionVars = assignmentDecisionPairs.filter { (assgn, _) -> assgn.all { fullAssignment[it.key] == it.value } }

            // Equality constraint is made up of a greater than constraint and a less than constraint
            if (conType == ConstraintType.EQ) {
                val gtConstraint = solver.makeConstraint()
                gtConstraint.setLb(partialState.reward - bigM)
                relevantDecisionVars.forEach { gtConstraint.setCoefficient(it.second, 1.0) }
                gtConstraint.setCoefficient(disjunctionVar, -bigM)

                val ltConstraint = solver.makeConstraint()
                ltConstraint.setUb(partialState.reward + bigM)
                relevantDecisionVars.forEach { ltConstraint.setCoefficient(it.second, 1.0) }
                ltConstraint.setCoefficient(disjunctionVar, bigM)
            } else {
                val gtConstraint = solver.makeConstraint()
                gtConstraint.setLb(partialState.reward + epsilon - bigM)
                relevantDecisionVars.forEach { gtConstraint.setCoefficient(it.second, 1.0) }
                gtConstraint.setCoefficient(disjunctionVar, -bigM)
            }
        }

        // At least one of disjunctions must be true
        val disjConstraint = solver.makeConstraint()
        disjunctionVars.forEach { disjConstraint.setCoefficient(it, 1.0) }
        disjConstraint.setLb(1.0)
    }
}

fun removeEmpty(rewardTables : List<RewardTable>) : List<RewardTable>{
    if(rewardTables.isEmpty()){
        throw IllegalArgumentException("Must provide non empty list of tables")
    }

    val filteredTables = rewardTables.filter { it.values.any { it > 0 } }
    if(filteredTables.isEmpty()){
        return rewardTables.take(1)
    }
    return filteredTables
}

fun compressSingleEntries(rewardTables : List<RewardTable>) : List<RewardTable> {
    val updatedRewardTables = rewardTables.toMutableList()
    for((rtIndex, rt) in updatedRewardTables.withIndex()){
        val (scope, rewards) = rt
        if(scope.size > 1){
            val nonZeroEntries = rewards.withIndex().filter{ (_, v) -> v > 0 }
            if(nonZeroEntries.size == 1){
                val (i, reward) = nonZeroEntries[0]
                val firstVariable = scope[0]
                val firstVarAssignment = indexToAssignment(i, scope)[0]
                val compressedValues = MutableList(firstVariable.domainSize){0.0}
                compressedValues[firstVarAssignment] = reward
                updatedRewardTables[rtIndex] = RewardTable(listOf(firstVariable), compressedValues)
            }
        }
    }
    return updatedRewardTables
}

fun reportSubsets(rewardTables : List<RewardTable>) : List<RewardTable>{
    val updatedRewardTables = rewardTables.toMutableList()
    val updatedRewardIterator = updatedRewardTables.listIterator()
    while(updatedRewardIterator.hasNext()){
        val rewardTable = updatedRewardIterator.next()
        val indexedSuperset = updatedRewardTables
            .withIndex()
            .find { (_, rt) -> rt.scope.containsAll(rewardTable.scope)
                && rt.scope.size > rewardTable.scope.size}

        if(indexedSuperset != null){
            val (i, superSet) = indexedSuperset
            updatedRewardTables[i] = plus(rewardTable, superSet)
            updatedRewardIterator.remove()
        }
    }
    return updatedRewardTables
}

fun plus(fromRT: RewardTable, toRT: RewardTable) : RewardTable {

    val extraVars = toRT.scope - fromRT.scope
    val r2StrideMap = toRT.scope.zip(getStrides(toRT.scope)).toMap()
    val indexAdditions = allAssignments(extraVars)
        .map{ it.entries.sumBy { (rv, v) -> r2StrideMap[rv]!! * v } }

    val summedValues = toRT.values.toMutableList()
    for((smallInd, reward) in fromRT.values.withIndex()){
        val projectedIndex = convertIndex(smallInd, fromRT.scope, toRT.scope)
        val newIndices = indexAdditions.map { projectedIndex + it }
        newIndices.forEach { summedValues[it] += reward }
    }

    return RewardTable(toRT.scope, summedValues)
}

fun assignmentAsString(assgn : RVAssignment) : String =
    assgn.map { (rv, v) ->
        val lcLetter = rv.toString().toLowerCase()
        if(v == 0) "¬$lcLetter" else lcLetter
    }.joinToString("")


fun unfactoredReward(rewardDom: List<RandomVariable>,
                     existsStates: List<AssignmentReward>): RewardTable? {

    System.loadLibrary("jniortools")
    val solver = Solver("Reward Function Solver")

    println("Reward Dom Size: ${rewardDom.size} : $rewardDom")
    val rewards = existsStates.map { it.reward }
    val rewardCSPSymbols = rewardToCSPSymbols(rewards, 1E-5)

    val numAssignments = rewardDom.productBy { it.domainSize }
    val numSymbols = (rewardCSPSymbols.keys.size - 1).toLong()
    val stateDecisionVars = solver.makeIntVarArray(numAssignments, 0, numSymbols)

    val equalityDisjuncts = constraintsToDisjunctions(rewardDom, rewardCSPSymbols, existsStates)
    equalityDisjuncts
        .map { (reward, states) -> states.map { solver.makeIsEqualCstVar(stateDecisionVars[it], reward) }}
        .forEach { solver.addConstraint(solver.makeSumGreaterOrEqual(it.toTypedArray(), 1)) }

    val displayParams = DefaultPhaseParameters()
    displayParams.display_level = DefaultPhaseParameters.NONE

    val decisionBuilder = solver.makeDefaultPhase(stateDecisionVars, displayParams)
    val CSPDomainToReward = rewardCSPSymbols.inverse()

    solver.newSearch(decisionBuilder)

    val result = if(solver.nextSolution())
        RewardTable(rewardDom, stateDecisionVars.map { CSPDomainToReward[it.value()]!! })
    else null

    solver.endSearch()
    solver.delete()

    return result
}


fun constraintsToDisjunctions(rewardDomain: List<RandomVariable>,
                              rewardToCSPDomain: BiMap<Double, Long>,
                              trials: List<AssignmentReward>): List<StateDisjunct> {
    val possibleAssignments = allAssignments(rewardDomain)
    return trials.map { assignRew ->
        val constraint = assignRew.assignment.filterKeys { rewardDomain.contains(it) }

        val matchingStates = possibleAssignments.withIndex()
            .filter { (_, assignment) -> constraint.all { assignment[it.key] == it.value} }
            .map{it.index}

        StateDisjunct(rewardToCSPDomain[assignRew.reward]!!, matchingStates)
    }
}

fun rewardToCSPSymbols(rewards: List<Double>, epsilon: Double): BiMap<Double, Long> {
    val rewardToCSPSymbol = rewards
        .sorted()
        .distinct()
        .flatMap { listOf(it, it + epsilon) }
        .withIndex()
        .associate{Pair(it.value, it.index.toLong())}
    return HashBiMap.create(rewardToCSPSymbol)
}

fun updateInfo(trials : List<Trial>, betterActionAdvice : List<BetterActionAdvice>, newTrial: Trial) : Pair<List<Trial>, List<BetterActionAdvice>>{
    // Update may be irrelevant
    if(trials.any{isSubset(newTrial.assignment, it.assignment) && it.reward == newTrial.reward }) {
        return Pair(trials, betterActionAdvice)
    }

    val filteredTrials = trials.filterNot { isSubset(it.assignment, newTrial.assignment) && it.reward == newTrial.reward }
    val filteredAdvice = betterActionAdvice.filterNot { isSubset(it.assignment, newTrial.assignment) && newTrial.reward > it.reward}
    return Pair(filteredTrials + newTrial, filteredAdvice)
}

fun updateInfo(trials : List<Trial>, betterActionAdvice: List<BetterActionAdvice>, newAdvice : BetterActionAdvice) : List<BetterActionAdvice>{
    if(trials.any{isSubset(newAdvice.assignment, it.assignment) && it.reward >= newAdvice.reward } ||
        betterActionAdvice.any { isSubset(newAdvice.assignment, it.assignment) && it.reward >= newAdvice.reward }) {
        return betterActionAdvice
    }

    val filteredAdvice = betterActionAdvice.filterNot { isSubset(it.assignment, newAdvice.assignment) && newAdvice.reward > it.reward}
    return filteredAdvice + newAdvice
}


/* Function isn't "truly" random. Intead, it aims for:
- Factors of size 1 just as likely as factors of size maxBound
- Each rv is used in at least one factor
- Once all variables have been used up, stop making factors
 */
fun randomFactorScopes(rvs : List<RandomVariable>, maxSize : Int) : List<List<RandomVariable>>{
    val allSubsets = powerSet(rvs, 1, maxSize)
    tailrec fun randomFactoringsRec(remainingVars : List<RandomVariable>, factorings : List<List<RandomVariable>>) : List<List<RandomVariable>> {
        if(remainingVars.isEmpty()){
            return factorings
        }
        val currentVar = remainingVars.first()
        val factorSize = random(1, Math.min(rvs.size, maxSize) + 1)
        val validSubsets = allSubsets.filter { it.size == factorSize && it.contains(currentVar) && !factorings.contains(it) }
        val chosenSubset = validSubsets.random()
        return randomFactoringsRec(remainingVars - chosenSubset, factorings + listOf(chosenSubset))
    }
    return randomFactoringsRec(shuffle(rvs), emptyList())
}

fun randomFactors(rvs : List<RandomVariable>, maxSize : Int) : List<RewardTable>{
    val factorScopes = randomFactorScopes(rvs, maxSize)
    return factorScopes.map{ RewardTable (it, randomList(0.0, 10.0, numAssignments((it))))}
}

data class PreferenceStatement(val conditionalAssignment : RVAssignment, val scope : RandomVariable, val prefOrdering : List<Int>)
