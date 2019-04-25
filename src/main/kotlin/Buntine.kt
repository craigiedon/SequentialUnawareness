import Utils.repeatNum
import Utils.sumInnerLnProbs
import com.fasterxml.jackson.annotation.JsonSubTypes
import com.fasterxml.jackson.annotation.JsonTypeInfo
import com.google.common.collect.Sets
import java.util.*

typealias PSet = Set<RandomVariable>
typealias DBNStruct = Map<RandomVariable, PSet>
typealias DirEdge = Pair<RandomVariable, RandomVariable>

data class SeqPInfo(val child : RandomVariable, val parentSet : PSet, val logProbability : Double, val counts : SequentialCountTable, val priorParams : Factor, private var nonZeroParentAssignments : Int? = null){
    init {
        if (priorParams.scope.first() != child || priorParams.scope.drop(1).toSet() != parentSet){
            throw IllegalArgumentException("Prior Param Scope ${priorParams.scope} does not match with PInfo variables ($child, $parentSet)")
        }
        if(counts.prevScope.toSet() != parentSet || counts.nextScope.toSet() != setOf(child)){
            throw IllegalArgumentException("Count Table scope (${counts.prevScope}, ${counts.nextScope}) does not match Seq PInfo scope ($parentSet, [$child])")
        }

        // Caching non-zero parents to speed up scoring later
        if(nonZeroParentAssignments == null){
            nonZeroParentAssignments = allAssignments(parentSet.toList())
                .asSequence()
                .filter { pAssgn -> count(pAssgn) > 0 }
                .count()
        }
    }

    fun nonZeroParentAssignments() : Int{
        return nonZeroParentAssignments!!
    }

    fun count(childVal : Int, prevAssignment: RVAssignment) =
        counts.getCount(prevAssignment, mapOf(child to childVal))

    fun count(parentAssignment: RVAssignment) =
        (0 until child.domainSize).sumByDouble { count(it, parentAssignment) }

    fun addTrial(seqTrial: SequentialTrial){
        if(count(seqTrial.prevState) == 0.0){
            nonZeroParentAssignments = nonZeroParentAssignments!! + 1
        }
        counts.updateCounts(seqTrial)
    }

    fun jointPrior(childVal : Int, parentAssgn : RVAssignment) : Double {
        // Complex index translation required because there are often 2 assignments to variable X: one in the previous time step (parent) and one in the current (child)
        val index = assignmentToIndex(listOf(childVal) + priorParams.scope.drop(1).map { parentAssgn[it]!! }, getStrides(priorParams.scope))
        return priorParams.values[index]
    }

    fun marginalPrior(parentAssgn : RVAssignment) =
        (0 until child.domainSize).sumByDouble { jointPrior(it, parentAssgn) }
}

data class DBNInfo(
    val reasonableParents : Map<RandomVariable, List<SeqPInfo>>,
    val bestPInfos : Map<RandomVariable, SeqPInfo>,
    val cptsITI : Map<RandomVariable, ProbTree>,
    val priorJointParams : Map<RandomVariable, DecisionTree<Factor>>,
    val dbn : DynamicBayesNet,
    val pSetPriors : Map<RandomVariable, LogPrior<PSet>>,
    val lastStructUpdate : Int
){
    val vocab : Set<RandomVariable> get() = dbn.keys
}

fun rawStructData(dbnInfo : DBNInfo) : Map<String, List<String>> =
    dbnInfo.bestPInfos
        .mapKeys { it.key.name }
        .mapValues { it.value.parentSet.map { it.name } }

data class ParentExtResult(val logPriors : Map<RandomVariable, LogPrior<PSet>>, val reasonableParents : Map<RandomVariable, List<PSet>>)

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = "type")
@JsonSubTypes(
    JsonSubTypes.Type(value = DegrisUpdater::class, name = "degris"),
    JsonSubTypes.Type(value = BuntineUpdater::class, name = "buntine"),
    JsonSubTypes.Type(value = NonConservativeUpdater::class, name="noncon")
)

interface DBNUpdater{
    fun addBeliefVariables(newVars : Set<RandomVariable>, timeOfUpdate: Int, seqTrials: List<SequentialTrial>, expertEv: List<DirEdge>, dbnInfo: DBNInfo) : DBNInfo
    fun initialDBNInfo(vocab : Set<RandomVariable>, expertEv: List<DirEdge>, timeStep : TimeStamp) : DBNInfo
    fun trialUpdate(seqTrial : SequentialTrial, trialHistory : List<SequentialTrial>, expertEv : List<DirEdge>, timeOfUpdate: TimeStamp, dbnInfo: DBNInfo) : DBNInfo
}

class DegrisUpdater(private val singleParentProb : Double, private val pseudoCountSize : Double) : DBNUpdater{
    override fun addBeliefVariables(newVars: Set<RandomVariable>, timeOfUpdate: Int, seqTrials: List<SequentialTrial>, expertEv: List<DirEdge>, dbnInfo: DBNInfo): DBNInfo {
        // Even though its a degris updater, we still let it conserve parameter information in the newly constructed param prior
        val oldVocabUpdatedJointPriorParams = dbnInfo.cptsITI
            .mapValues { (rv, cpt) -> convertToJointProbTree(rv, cpt.first, dbnInfo.priorJointParams[rv]!!, pseudoCountSize) }

        val newVarJointParamPriors = unifStartIDTransJoint(newVars)
        val finalJointParamPrior = oldVocabUpdatedJointPriorParams + newVarJointParamPriors

        val updatedVocab = dbnInfo.vocab + newVars
        val bestParents = updatedVocab.associate { Pair(it, updatedVocab) }

        val cptsITI = initialCPTsITI(updatedVocab, bestParents, finalJointParamPrior, pseudoCountSize)
        val dbn = cptsITI
            .mapValues { (rv, cpt) -> convertToCPT(rv, cpt.first, finalJointParamPrior[rv]!!, pseudoCountSize) }

        return DBNInfo(emptyMap(), emptyMap(), cptsITI, finalJointParamPrior, dbn, emptyMap(), timeOfUpdate)
    }

    override fun initialDBNInfo(vocab: Set<RandomVariable>, expertEv: List<DirEdge>, timeStep: TimeStamp): DBNInfo {
        val priorJointParams = unifStartIDTransJoint(vocab)

        // All vocabulary is allowed in every CPT: Structure learning is not encapsulated, but rather explicit in DT learning
        val cptsITI = initialCPTsITI(vocab, vocab.associate { Pair(it, vocab.toSet()) },  priorJointParams, pseudoCountSize)
        val dbn = convertToCPT(cptsITI, priorJointParams, pseudoCountSize)

        return DBNInfo(emptyMap(), emptyMap(), cptsITI, priorJointParams, dbn, emptyMap(), timeStep)
    }

    override fun trialUpdate(seqTrial: SequentialTrial, trialHistory: List<SequentialTrial>, expertEv: List<DirEdge>, timeOfUpdate: TimeStamp, dbnInfo: DBNInfo): DBNInfo {
        val cptsITI = dbnInfo.cptsITI.mapValues { (_, cpt) -> Pair(incrementalUpdate(cpt.first, listOf(seqTrial), cpt.second), cpt.second) }
        val dbn = convertToCPT(cptsITI, dbnInfo.priorJointParams, pseudoCountSize)

        return dbnInfo.copy(dbn = dbn, cptsITI = cptsITI)
    }
}

class NonConservativeUpdater(private val singleParentProb : Double, private val pseudoCountSize: Double, private val maxParents : Int) : DBNUpdater {
    override fun addBeliefVariables(newVars: Set<RandomVariable>, timeOfUpdate: Int, seqTrials: List<SequentialTrial>, expertEv: List<DirEdge>, dbnInfo: DBNInfo): DBNInfo {
        return initialDBNInfo(dbnInfo.vocab + newVars, expertEv, timeOfUpdate)
    }

    override fun initialDBNInfo(vocab: Set<RandomVariable>, expertEv: List<DirEdge>, timeStep: TimeStamp): DBNInfo {
        val priorJointParams = unifStartIDTransJoint(vocab)
        val pSetPriors = initialPSetPriors(vocab, singleParentProb)
        val reasonableParents = structuralUpdate(vocab, emptyList(), expertEv, pSetPriors, 0.0, priorJointParams, maxParents)
        val bestPInfos = bestParents(reasonableParents)
        val cptsITI = initialCPTsITI(vocab, bestPInfos.mapValues { it.value.parentSet }, priorJointParams, pseudoCountSize)
        val dbn = convertToCPT(cptsITI, priorJointParams, pseudoCountSize)

        return DBNInfo(reasonableParents, bestPInfos, cptsITI, priorJointParams, dbn, pSetPriors, timeStep)
    }

    override fun trialUpdate(seqTrial : SequentialTrial, trialHistory : List<SequentialTrial>, expertEv : List<DirEdge>, timeOfUpdate: TimeStamp, dbnInfo: DBNInfo) : DBNInfo{
        val newReasonableParents = dbnInfo.reasonableParents.mapValues { (rv, rps) -> parameterUpdate(rv, rps, seqTrial, expertEv, dbnInfo.pSetPriors[rv]!!) }
        val bestParents = bestParents(newReasonableParents)

        // Im not 100% sold on this coupling of cpt tree updates with buntine updater (which should only really be about structure i think...)
        val cptsITI = dbnInfo.cptsITI.mapValues { (rv, cpt) ->
            val (newVocabDT, newConfig) = changeAllowedVocab(cpt, bestParents[rv]!!.parentSet)
            Pair(incrementalUpdate(newVocabDT, listOf(seqTrial), newConfig), newConfig)
        }

        val dbn = convertToCPT(cptsITI, dbnInfo.priorJointParams, pseudoCountSize)

        return dbnInfo.copy(reasonableParents = newReasonableParents, bestPInfos = bestParents, dbn = dbn, cptsITI = cptsITI)
    }
}

class BuntineUpdater(private val structUpdateInterval : Int,
                     private val maxParents : Int,
                     private val aliveThresh: Double,
                     private val pseudoCountSize : Double,
                     private val singleParentProb : Double) : DBNUpdater {

    override fun addBeliefVariables(newVars: Set<RandomVariable>, timeOfUpdate: Int, seqTrials: List<SequentialTrial>, expertEv: List<DirEdge>, dbnInfo: DBNInfo): DBNInfo {
        val oldVocab = dbnInfo.vocab
        val oldVocabReasonablePs = structuralUpdate(oldVocab, seqTrials, expertEv, dbnInfo.pSetPriors, aliveThresh, dbnInfo.priorJointParams, maxParents)
        val oldVocabBest = bestParents(oldVocabReasonablePs)
        val oldVocabCPTs = dbnInfo.cptsITI.mapValues { (rv, cpt) -> changeAllowedVocab(cpt, oldVocabBest[rv]!!.parentSet) }
        val oldLogProbs = oldVocabReasonablePs.mapValues { (_, rps) -> rps.associate { Pair(it.parentSet, it.logProbability) } }

        // Also, remake your dbn at this point (decision trees and all):
        val (oldVocabUpdatedlogPriors, oldVocabUpdatedPSets) = posteriorToPrior(oldLogProbs, oldVocab, newVars, singleParentProb, aliveThresh, maxParents)
        val oldVocabUpdatedJointPriorParams = oldVocabCPTs.mapValues { (rv, cpt) -> convertToJointProbTree(rv, cpt.first, dbnInfo.priorJointParams[rv]!!, pseudoCountSize) }
        val oldVocabUpdatedRPs = oldVocabUpdatedPSets.mapValues { (rv, pSets) -> pSets.map{ createPInfo(rv, it, emptyList(), oldVocabUpdatedlogPriors[rv]!!, oldVocabUpdatedJointPriorParams[rv]!!, expertEv) } }

        // Create new param priors for new vocab
        val newVarJointParamPriors = unifStartIDTransJoint(newVars)
        val newVarPriorFuncs = newVars.associate{ Pair(it, minParentsPrior(it, oldVocab + newVars, singleParentProb)) }
        val newVarReasonablePs = structuralUpdate(oldVocab + newVars, emptyList(), expertEv, newVarPriorFuncs, aliveThresh, newVarJointParamPriors, maxParents)

        // Combine old and new vocab
        val finalReasonableParents = oldVocabUpdatedRPs + newVarReasonablePs
        val finalLogPriors = oldVocabUpdatedlogPriors + newVarPriorFuncs
        val finalJointParamPrior = oldVocabUpdatedJointPriorParams + newVarJointParamPriors

        val bestParents = bestParents(finalReasonableParents)
        val cptsITI = initialCPTsITI(oldVocab + newVars, bestParents.mapValues { it.value.parentSet }, finalJointParamPrior, pseudoCountSize)
        val dbn = cptsITI.mapValues { (rv, cpt) -> convertToCPT(rv, cpt.first, finalJointParamPrior[rv]!!, pseudoCountSize) }

        return DBNInfo(finalReasonableParents, bestParents, cptsITI, finalJointParamPrior, dbn, finalLogPriors, timeOfUpdate)
    }

    override fun initialDBNInfo(vocab: Set<RandomVariable>, expertEv: List<DirEdge>, timeStep: TimeStamp): DBNInfo {
        val priorJointParams = unifStartIDTransJoint(vocab)
        val pSetPriors = initialPSetPriors(vocab, singleParentProb)
        val reasonableParents = structuralUpdate(vocab, emptyList(), expertEv, pSetPriors, aliveThresh, priorJointParams, maxParents)
        val bestPInfos = bestParents(reasonableParents)
        val cptsITI = initialCPTsITI(vocab, bestPInfos.mapValues { it.value.parentSet }, priorJointParams, pseudoCountSize)
        val dbn = convertToCPT(cptsITI, priorJointParams, pseudoCountSize)

        return DBNInfo(reasonableParents, bestPInfos, cptsITI, priorJointParams, dbn, pSetPriors, timeStep)
    }

    override fun trialUpdate(seqTrial : SequentialTrial, trialHistory : List<SequentialTrial>, expertEv : List<DirEdge>, timeOfUpdate: TimeStamp, dbnInfo: DBNInfo) : DBNInfo{
        var newReasonableParents = dbnInfo.reasonableParents.mapValues { (rv, rps) -> parameterUpdate(rv, rps, seqTrial, expertEv, dbnInfo.pSetPriors[rv]!!) }
        var lastStructUpdate = dbnInfo.lastStructUpdate

        if(timeOfUpdate - dbnInfo.lastStructUpdate > structUpdateInterval){
            lastStructUpdate = timeOfUpdate
            newReasonableParents = newReasonableParents
                .mapValues { (rv, _) -> structuralUpdate(rv, dbnInfo.pSetPriors[rv]!!, dbnInfo.vocab, trialHistory, expertEv, dbnInfo.priorJointParams[rv]!!, aliveThresh, maxParents) }
        }
        val bestParents = bestParents(newReasonableParents)

        // Im not 100% sold on this coupling of cpt tree updates with buntine updater (which should only really be about structure i think...)
        val cptsITI = dbnInfo.cptsITI
            .mapValues { (rv, cpt) ->
                val (newVocabDT, newConfig) = changeAllowedVocab(cpt, bestParents[rv]!!.parentSet)
                Pair(incrementalUpdate(newVocabDT, listOf(seqTrial), newConfig), newConfig)
            }

        val dbn = convertToCPT(cptsITI, dbnInfo.priorJointParams, pseudoCountSize)

        return dbnInfo.copy(reasonableParents = newReasonableParents, bestPInfos = bestParents, lastStructUpdate = lastStructUpdate, dbn = dbn, cptsITI = cptsITI)
    }
}



fun parameterUpdate(child : RandomVariable,
                    reasonableParents :  List<SeqPInfo>,
                    seqTrial : SequentialTrial,
                    expertEvidence : List<DirEdge>,
                    structuralPrior: LogPrior<PSet>) : List<SeqPInfo>{

    val pseudoCountSize = 1.0
    val newReasonableParentSets = reasonableParents.map { pInfo ->
        //Sample counts are updated incrementally here
        pInfo.addTrial(seqTrial)

        val parentAssgn = seqTrial.prevState.filterKeys { it in pInfo.parentSet }
        val childVal = seqTrial.currentState[child]!!

        val countJoint = pInfo.count(childVal, parentAssgn)
        val countMarginal = pInfo.count(parentAssgn)

        // With BDsScore, should only need to do full rescoring if an assignment was previously 0
        val newLogProb : Double
        if(countMarginal.toInt() <= 1){
            newLogProb = structuralPrior(pInfo.parentSet) + BDsScore(pInfo.child, pInfo.parentSet, pInfo.counts, pseudoCountSize)
        }
        else{
            //Update Structural Probability
            val alphaJoint = pseudoCountSize / (pInfo.nonZeroParentAssignments() * pInfo.child.domainSize)
            val alphaMarginal = pseudoCountSize / (pInfo.nonZeroParentAssignments())
            newLogProb = pInfo.logProbability + Math.log(countJoint + alphaJoint - 1) - Math.log(countMarginal + alphaMarginal - 1)
        }


        pInfo.copy(logProbability = newLogProb)
    }

    return logNorm(newReasonableParentSets.filter{ !violatesEvidence(child, it.parentSet, expertEvidence)})
}

fun violatesEvidence(child : RandomVariable, parentSet : PSet, parentEvidence: List<DirEdge>) =
    parentEvidence.any { (pEv, cEv) ->
        val requiredParentViolation = cEv == child && pEv !in parentSet
        val asymmetryViolation = pEv == child && cEv in parentSet
        requiredParentViolation || asymmetryViolation
    }

fun logNorm(parentSetInfos: List<SeqPInfo>) : List<SeqPInfo> {
    if(parentSetInfos.isEmpty()){
        return emptyList()
    }
    val logTotal = sumInnerLnProbs(parentSetInfos.map { it.logProbability })
    return parentSetInfos.map { p -> p.copy(logProbability = p.logProbability - logTotal)}
}

data class LatticeNode<V> constructor(val item : V, val parents : MutableList<LatticeNode<V>> = ArrayList()){
    val children = mutableListOf<LatticeNode<V>>()
    init {
        parents.forEach { it.children.add(this) }
    }
}

typealias PSetNode = LatticeNode<PSet>
typealias LogPrior<T> = (T) -> Double

fun createPInfo(child : RandomVariable, pSet: PSet, trials : List<SequentialTrial>, logPrior: LogPrior<PSet>, priorJointParams: DecisionTree<Factor>, expertEv : List<DirEdge>) : SeqPInfo{
    val orderedParents = pSet.toList()

    val counts = SequentialCountTable(orderedParents, listOf(child), trials)
    val priorParams = priorParamTable(child, pSet, priorJointParams)

    val score = when(violatesEvidence(child, pSet, expertEv)){
        true -> Math.log(0.0)
        false -> logPrior(pSet) + BDsScore(child, pSet, counts, 1.0)
    }

    if(score.isNaN() || score.isInfinite()){
        val priorPart = logPrior(pSet)
        val likelihoodPart = BDsScore(child, pSet, counts, 1.0)
        throw IllegalStateException("Score is not a number!")
    }

    return SeqPInfo(child, pSet, score, counts, priorParams)
}

fun emptyCounts(child : RandomVariable, pSet : PSet) =
    allAssignments(pSet.toList())
        .associate { Pair(it, repeatNum(0, child.domainSize).toMutableList()) }

fun priorParamTable(rv : RandomVariable, pSet: PSet, dt : DecisionTree<Factor>) : Factor {
    val orderedPSet = pSet.toList()
    val jointValues = allAssignments(orderedPSet)
        .flatMap { pAssign -> jointQuery(pAssign, dt).values }
    return Factor(listOf(rv) + orderedPSet, jointValues)
}

fun <V> siblings(node : LatticeNode<V>) : List<LatticeNode<V>> =
    node.parents.flatMap { p -> p.children.filter { it.item != node.item } }

fun structuralUpdate(vocab: Set<RandomVariable>, trials: List<SequentialTrial>,
                     expertEv: List<DirEdge>,
                     priors: Map<RandomVariable, LogPrior<PSet>>,
                     aliveThresh: Double,
                     jointPriorParams: Map<RandomVariable, DecisionTree<Factor>>,
                     maxParents: Int): Map<RandomVariable, List<SeqPInfo>> {
    return priors.mapValues { (rv, prior) ->
        structuralUpdate(rv, prior, vocab, trials, expertEv, jointPriorParams[rv]!!, aliveThresh, maxParents)
    }
}

fun structuralUpdate(X : RandomVariable,
                     logPrior: LogPrior<PSet>,
                     vocab: Collection<RandomVariable>,
                     trials: List<SequentialTrial>,
                     expertEv: List<DirEdge>,
                     jointPriorParams: DecisionTree<Factor>,
                     aliveThresh: Double,
                     maxParents : Int): List<SeqPInfo> {

    val openList = LinkedList<PSetNode>()
    var aliveList = emptyList<PSetNode>()

    val lattice = HashMap<PSet, PSetNode>()
    val pSetInfos = HashMap<PSet, SeqPInfo>()

    val minNode = PSetNode(expertEv
        .asSequence()
        .filter { (_,c) -> c == X }
        .map { (p, _) -> p }
        .toSet()
    )

    lattice[minNode.item] = minNode
    pSetInfos[minNode.item] = createPInfo(X, minNode.item, trials, logPrior, jointPriorParams, expertEv)
    var bestScore = pSetInfos[minNode.item]!!.logProbability
    aliveList += minNode

    val minNodeExtensions = vocab
        .filter { it !in minNode.item }
        .map { PSetNode(minNode.item + it, mutableListOf(minNode)) }

    openList.addAll(minNodeExtensions)

    while(openList.isNotEmpty()){
        val node = openList.poll()
        val oldBestScore = bestScore

        if(node.item !in pSetInfos){
            val pInfo = createPInfo(X, node.item, trials, logPrior, jointPriorParams, expertEv)
            pSetInfos[node.item] = pInfo
            if(pInfo.logProbability > Math.log(aliveThresh)  + bestScore){
                aliveList += node
                for(sibling in siblings(node)){
                    val pSetUnion = node.item + sibling.item
                    if(pSetUnion.size <= maxParents && pSetUnion !in lattice){
                        val unionNode = PSetNode(pSetUnion, mutableListOf(node, sibling))
                        lattice[pSetUnion] = unionNode
                        openList.add(unionNode)
                    }
                }
            }
            bestScore = maxOf(pInfo.logProbability, bestScore)

            if(oldBestScore < bestScore){
                aliveList = revisedAliveList(aliveThresh, aliveList, pSetInfos)
            }
        }
    }

    return logNorm(aliveList.map { pSetInfos[it.item]!!})
}

fun revisedAliveList(aliveThresh: Double, aliveList: List<PSetNode>, pSetInfos: Map<PSet, SeqPInfo>) : List<PSetNode> {
    if(aliveList.isEmpty()){
        throw IllegalStateException("No parent sets deemed reasonable!")
    }
    val bestScore = pSetInfos.values.map { it.logProbability }.max()!!
    return aliveList.filter { isAlive(pSetInfos[it.item]!!.logProbability, aliveThresh, bestScore) }
}

fun isAlive(logProbability: Double, aliveThresh: Double, logBest: Double) =
    logProbability > Math.log(aliveThresh) + logBest


@JvmName("BestParentsPInfo")
fun bestParents(parentChoices : Map<RandomVariable, List<SeqPInfo>>) : Map<RandomVariable, SeqPInfo> =
    parentChoices.mapValues{ (_, reasonableParents) -> reasonableParents.maxBy { it.logProbability }!!}


fun posteriorToPrior(rpLogProbs: Map<RandomVariable, Map<PSet, Double>>, oldVocab : Set<RandomVariable>, extraVocab : Set<RandomVariable>,
                     singleParentProb : Double, aliveThresh : Double, maxParents : Int) : ParentExtResult {

    val inMix = 0.99

    val priorMaps = rpLogProbs
        .mapValues { (_, rps) -> rps
            .flatMap { (oldPSet, oldLogProb) ->
                Sets.powerSet(extraVocab)
                    .map { extraSubset ->
                        val pSet = oldPSet + extraSubset
                        val logProb = extraSubset.size * Math.log(singleParentProb) + (extraVocab.size - extraSubset.size) * Math.log(1 - singleParentProb) + oldLogProb
                        Pair(pSet, logProb)
                    }
            }
            .asSequence()
            .filter { it.first.size <= maxParents }
            .associate{ it }
        }


    val priorFuncs = priorMaps.mapValues{ (rv, priorMap) ->
        val minPrior = minParentsPrior(rv, oldVocab + extraVocab, singleParentProb)
        val mapPrior = { pSet : PSet -> mapPrior(pSet, priorMap)}
        val mixPrior = { pSet : PSet -> mixedPrior(pSet, inMix, mapPrior, minPrior) }
        mixPrior
    }

    val bestPSetScores = priorMaps.mapValues { (_, rps) -> rps.values.max() }

    val newReasonable = priorMaps
        .mapValues { (rv, rps) -> rps
            .filterValues { it >= bestPSetScores[rv]!! + Math.log(aliveThresh) }
            .map { it.key }
        }

    return ParentExtResult(priorFuncs, newReasonable)
}

fun minParentsPrior(child : RandomVariable, vocab : Set<RandomVariable>, extraParentCost : Double) : LogPrior<PSet>{
    return { pSet -> vocab.sumByDouble { rv ->
        Math.log(
            when{
                rv == child -> 0.5 // It is more likely that variable has itself as parent than some other variable
                (rv in pSet) -> extraParentCost
                else -> 1.0 - extraParentCost
            }
        )
    }}
}

fun mapPrior(pSet: PSet, priorMap : Map<PSet, Double>) =
    if(pSet in priorMap) priorMap[pSet]!! else Math.log(0.0)


fun mixedPrior(pSet : PSet, inMix : Double, inPrior : LogPrior<PSet>, outPrior : LogPrior<PSet>) : Double {
    val inVal = Math.log(inMix) + inPrior(pSet)
    val outVal = Math.log(1 - inMix) + outPrior(pSet)
    return sumInnerLnProbs(listOf(inVal, outVal))
}
