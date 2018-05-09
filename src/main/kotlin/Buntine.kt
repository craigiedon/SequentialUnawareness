import Utils.repeatNum
import Utils.sumInnerLnProbs
import com.fasterxml.jackson.annotation.JsonSubTypes
import com.fasterxml.jackson.annotation.JsonTypeInfo
import com.google.common.collect.Sets
import java.util.*

typealias PSet = Set<RandomVariable>
typealias DBNStruct = Map<RandomVariable, PSet>
typealias DirEdge = Pair<RandomVariable, RandomVariable>

data class SeqPInfo(val child : RandomVariable, val parentSet : PSet, val logProbability : Double, val counts : Map<RVAssignment, MutableList<Int>>, val priorParams : Map<RVAssignment, Factor>){
    fun count(childVal : Int, parentAssignment: RVAssignment) =
        counts[parentAssignment]!![childVal]

    fun count(parentAssignment: RVAssignment) =
        counts[parentAssignment]!!.sum()

    fun addTrial(seqTrial: SequentialTrial){
        val parentAssignment = seqTrial.prevState.filterKeys { it in parentSet }
        val childVal = seqTrial.currentState[child]!!
        counts[parentAssignment]!![childVal] += 1
    }

    fun jointPrior(childVal : Int, parentAssgn : RVAssignment) =
        priorParams[parentAssgn]!!.values[childVal]

    fun marginalPrior(parentAssgn : RVAssignment) =
        priorParams[parentAssgn]!!.values.sumByDouble { it }
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
    JsonSubTypes.Type(value = BuntineUpdater::class, name = "buntine")
)
interface DBNUpdater{
    fun addBeliefVariables(newVars : Set<RandomVariable>, timeOfUpdate: Int, seqTrials: List<SequentialTrial>, expertEv: List<DirEdge>, dbnInfo: DBNInfo) : DBNInfo
    fun initialDBNInfo(vocab : Set<RandomVariable>, expertEv: List<DirEdge>, timeStep : TimeStamp) : DBNInfo
    fun trialUpdate(seqTrial : SequentialTrial, trialHistory : List<SequentialTrial>, expertEv : List<DirEdge>, timeOfUpdate: TimeStamp, dbnInfo: DBNInfo) : DBNInfo
}

class DegrisUpdater(val pseudoCountSize : Double) : DBNUpdater{
    override fun initialDBNInfo(vocab: Set<RandomVariable>, expertEv: List<DirEdge>, timeStep: TimeStamp): DBNInfo {
        val priorJointParams = unifStartIDTransJoint(vocab)

        val cptsITI = initialCPTsITI(vocab, vocab.associate { Pair(it, vocab.toSet()) },  pseudoCountSize)
        val dbn = cptsITIToDBN(cptsITI, priorJointParams, pseudoCountSize)

        return DBNInfo(emptyMap(), emptyMap(), cptsITI, priorJointParams, dbn, emptyMap(), timeStep)
    }

    override fun trialUpdate(seqTrial: SequentialTrial, trialHistory: List<SequentialTrial>, expertEv: List<DirEdge>, timeOfUpdate: TimeStamp, dbnInfo: DBNInfo): DBNInfo {
        val cptsITI = dbnInfo.cptsITI.mapValues { (_, cpt) -> Pair(incrementalUpdate(cpt.first, listOf(seqTrial), cpt.second), cpt.second) }
        val dbn = cptsITIToDBN(cptsITI, dbnInfo.priorJointParams, pseudoCountSize)

        return dbnInfo.copy(dbn = dbn, cptsITI = cptsITI)
    }

    override fun addBeliefVariables(newVars: Set<RandomVariable>, timeOfUpdate: Int, seqTrials: List<SequentialTrial>, expertEv: List<DirEdge>, dbnInfo: DBNInfo): DBNInfo {
        val oldVocabUpdatedJointPriorParams = dbnInfo.cptsITI.mapValues { (rv, cpt) -> convertToJointProbTree(rv, cpt.first, dbnInfo.priorJointParams[rv]!!, pseudoCountSize) }
        val newVarJointParamPriors = unifStartIDTransJoint(newVars)
        val finalJointParamPrior = oldVocabUpdatedJointPriorParams + newVarJointParamPriors

        val updatedVocab = dbnInfo.vocab + newVars
        val bestParents = updatedVocab.associate { Pair(it, updatedVocab) }

        val cptsITI = initialCPTsITI(updatedVocab, bestParents, pseudoCountSize)
        val dbn = cptsITI.mapValues { (rv, cpt) -> convertToCPT(rv, cpt.first, oldVocabUpdatedJointPriorParams[rv]!!, pseudoCountSize) }

        return DBNInfo(emptyMap(), emptyMap(), cptsITI, finalJointParamPrior, dbn, emptyMap(), timeOfUpdate)
    }

}

class BuntineUpdater(private val structUpdateInterval : Int,
                     private val aliveThresh: Double,
                     private val pseudoCountSize : Double,
                     private val singleParentProb : Double) : DBNUpdater {

    override fun addBeliefVariables(newVars: Set<RandomVariable>, timeOfUpdate: Int, seqTrials: List<SequentialTrial>, expertEv: List<DirEdge>, dbnInfo: DBNInfo): DBNInfo {
        val oldVocab = dbnInfo.vocab
        val oldVocabReasonablePs = structuralUpdate(oldVocab, seqTrials, expertEv, dbnInfo.pSetPriors, aliveThresh, dbnInfo.priorJointParams, pseudoCountSize)
        val oldVocabBest = bestParents(oldVocabReasonablePs)
        val oldVocabCPTs = dbnInfo.cptsITI.mapValues { (rv, cpt) -> changeAllowedVocab(cpt, oldVocabBest[rv]!!.parentSet) }
        val oldLogProbs = oldVocabReasonablePs.mapValues { (_, rps) -> rps.associate { Pair(it.parentSet, it.logProbability) } }

        // Also, remake your dbn at this point (decision trees and all):
        val (oldVocabUpdatedlogPriors, oldVocabUpdatedPSets) = posteriorToPrior(oldLogProbs, oldVocab, newVars, singleParentProb, aliveThresh)
        val oldVocabUpdatedJointPriorParams = oldVocabCPTs.mapValues { (rv, cpt) -> convertToJointProbTree(rv, cpt.first, dbnInfo.priorJointParams[rv]!!, pseudoCountSize) }
        val oldVocabUpdatedRPs = oldVocabUpdatedPSets.mapValues { (rv, pSets) -> pSets.map{ createPInfo(rv, it, emptyList(), oldVocabUpdatedlogPriors[rv]!!, oldVocabUpdatedJointPriorParams[rv]!!, pseudoCountSize, expertEv) } }

        // Create new param priors for new vocab
        val newVarJointParamPriors = unifStartIDTransJoint(newVars)
        val newVarPriorFuncs = newVars.associate{ Pair(it, minParentsPrior(it, oldVocab + newVars, singleParentProb)) }
        val newVarReasonablePs = structuralUpdate(oldVocab + newVars, emptyList(), expertEv, newVarPriorFuncs, aliveThresh, newVarJointParamPriors, pseudoCountSize)

        // Combine old and new vocab
        val finalReasonableParents = oldVocabUpdatedRPs + newVarReasonablePs
        val finalLogPriors = oldVocabUpdatedlogPriors + newVarPriorFuncs
        val finalJointParamPrior = oldVocabUpdatedJointPriorParams + newVarJointParamPriors

        val bestParents = bestParents(finalReasonableParents)
        val cptsITI = initialCPTsITI(oldVocab + newVars, bestParents.mapValues { it.value.parentSet }, pseudoCountSize)
        val dbn = cptsITI.mapValues { (rv, cpt) -> convertToCPT(rv, cpt.first, finalJointParamPrior[rv]!!, pseudoCountSize) }

        return DBNInfo(finalReasonableParents, bestParents, cptsITI, finalJointParamPrior, dbn, finalLogPriors, timeOfUpdate)
    }

    override fun initialDBNInfo(vocab: Set<RandomVariable>, expertEv: List<DirEdge>, timeStep: TimeStamp): DBNInfo {
        val priorJointParams = unifStartIDTransJoint(vocab)
        val pSetPriors = initialPSetPriors(vocab, singleParentProb)
        val reasonableParents = structuralUpdate(vocab, emptyList(), expertEv, pSetPriors, aliveThresh, priorJointParams, pseudoCountSize)
        val bestPInfos = bestParents(reasonableParents)
        val cptsITI = initialCPTsITI(vocab, bestPInfos.mapValues { it.value.parentSet }, pseudoCountSize)
        val dbn = cptsITIToDBN(cptsITI, priorJointParams, pseudoCountSize)

        return DBNInfo(reasonableParents, bestPInfos, cptsITI, priorJointParams, dbn, pSetPriors, timeStep)
    }

    override fun trialUpdate(seqTrial : SequentialTrial, trialHistory : List<SequentialTrial>, expertEv : List<DirEdge>, timeOfUpdate: TimeStamp, dbnInfo: DBNInfo) : DBNInfo{
        var newReasonableParents = dbnInfo.reasonableParents.mapValues { (rv, rps) -> parameterUpdate(rv, rps, seqTrial, expertEv, dbnInfo.pSetPriors[rv]!!, pseudoCountSize ) }
        var lastStructUpdate = dbnInfo.lastStructUpdate

        if(timeOfUpdate - dbnInfo.lastStructUpdate > structUpdateInterval){
            lastStructUpdate = timeOfUpdate
            newReasonableParents = newReasonableParents.mapValues { (rv, _) -> structuralUpdate(rv, dbnInfo.pSetPriors[rv]!!, dbnInfo.vocab, trialHistory, expertEv, dbnInfo.priorJointParams[rv]!!, aliveThresh, pseudoCountSize) }
        }
        val bestParents = bestParents(newReasonableParents)

        // Im not 100% sold on this coupling of cpt tree updates with buntine updater (which should only really be about structure i think...)
        val cptsITI = dbnInfo.cptsITI.mapValues { (rv, cpt) ->
            val (newVocabDT, newConfig) = changeAllowedVocab(cpt, bestParents[rv]!!.parentSet)
            Pair(incrementalUpdate(newVocabDT, listOf(seqTrial), newConfig), newConfig)
        }

        val dbn = cptsITIToDBN(cptsITI, dbnInfo.priorJointParams, pseudoCountSize)

        return dbnInfo.copy(reasonableParents = newReasonableParents, bestPInfos = bestParents, lastStructUpdate = lastStructUpdate, dbn = dbn, cptsITI = cptsITI)
    }
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


fun parameterUpdate(child : RandomVariable,
                    reasonableParents :  List<SeqPInfo>,
                    seqTrial : SequentialTrial,
                    expertEvidence : List<DirEdge>,
                    structuralPrior: LogPrior<PSet>,
                    alphaTotal: Double) : List<SeqPInfo>{
    val newReasonableParentSets = reasonableParents.map { pInfo ->
        //Sample counts are updated incrementally here
        pInfo.addTrial(seqTrial)

        /*
        val parentAssgn = seqTrial.prevState.filterKeys { it in pInfo.parentSet }
        val childVal = seqTrial.currentState[child]!!

        val countJoint = pInfo.count(childVal, parentAssgn)
        val countMarginal = pInfo.count(parentAssgn)

        val alphaJoint = alphaTotal * pInfo.jointPrior(childVal, parentAssgn)
        val alphaMarginal = alphaTotal * pInfo.marginalPrior(parentAssgn)

        //Update Structural Probability
        val newLogProb = pInfo.logProbability + Math.log(countJoint + alphaJoint - 1) - Math.log(countMarginal + alphaMarginal - 1)
        */
        val newLogProb = structuralPrior(pInfo.parentSet) + BDsScore(pInfo.child, pInfo.parentSet, pInfo.counts, alphaTotal)
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

fun createPInfo(child : RandomVariable, pSet: PSet, trials : List<SequentialTrial>, logPrior: LogPrior<PSet>, priorJointParams: DecisionTree<Factor>, priorSampleSize : Double, expertEv : List<DirEdge>) : SeqPInfo{
    val counts = emptyCounts(child, pSet)

    for((prevState, _, currentState) in trials){
        val pAssgn = prevState.filterKeys { it in pSet }
        val childAssign = currentState[child]!!
        counts[pAssgn]!![childAssign] += 1
    }

    val priorParams = priorParamTable(pSet, priorJointParams)

    val score = when(violatesEvidence(child, pSet, expertEv)){
        true -> Math.log(0.0)
        false -> logPrior(pSet) + BDsScore(child, pSet, counts, priorSampleSize)
    }

    if(score.isNaN() || score.isInfinite()){
        throw IllegalStateException("Score is not a number!")
    }

    return SeqPInfo(child, pSet, score, counts, priorParams)
}

fun emptyCounts(child : RandomVariable, pSet : PSet) =
    allAssignments(pSet.toList())
        .associate { Pair(it, repeatNum(0, child.domainSize).toMutableList()) }

fun priorParamTable(pSet: PSet, dt : DecisionTree<Factor>) : Map<RVAssignment, Factor> =
    allAssignments(pSet.toList()).associate { pAssign ->
        // You should probably delete this and just replace with a version of joint query that can handle parent assignments
        val testCombo = pAssign
            .mapKeys { Pair(it.key, it.value) }
            .mapValues { true }
        Pair(pAssign, jointQuery(testCombo, dt))
    }

fun <V> siblings(node : LatticeNode<V>) : List<LatticeNode<V>> =
    node.parents.flatMap { p -> p.children.filter { it.item != node.item } }

fun structuralUpdate(vocab: Set<RandomVariable>, trials: List<SequentialTrial>, expertEv: List<DirEdge>, priors: Map<RandomVariable, LogPrior<PSet>>, aliveThresh: Double, jointPriorParams: Map<RandomVariable, DecisionTree<Factor>>, alphaStrength: Double): Map<RandomVariable, List<SeqPInfo>> {
    return priors.mapValues { (rv, prior) ->
        structuralUpdate(rv, prior, vocab, trials, expertEv, jointPriorParams[rv]!!, aliveThresh, alphaStrength)
    }
}

fun structuralUpdate(X : RandomVariable,
                     logPrior: LogPrior<PSet>,
                     vocab: Collection<RandomVariable>,
                     trials: List<SequentialTrial>,
                     expertEv: List<DirEdge>,
                     jointPriorParams: DecisionTree<Factor>,
                     aliveThresh: Double,
                     alphaStrength: Double): List<SeqPInfo> {

    val openList = LinkedList<PSetNode>()
    var aliveList = emptyList<PSetNode>()

    val lattice = HashMap<PSet, PSetNode>()
    val pSetInfos = HashMap<PSet, SeqPInfo>()

    val minNode = PSetNode(expertEv.filter { (_,c) -> c == X }.map { (p, _) -> p }.toSet())
    lattice[minNode.item] = minNode
    pSetInfos[minNode.item] = createPInfo(X, minNode.item, trials, logPrior, jointPriorParams, alphaStrength, expertEv)
    var bestScore = pSetInfos[minNode.item]!!.logProbability
    aliveList += minNode

    val minNodeExtensions = vocab.filter { it !in minNode.item }.map { PSetNode(minNode.item + it, mutableListOf(minNode)) }
    openList.addAll(minNodeExtensions)

    while(openList.isNotEmpty()){
        val node = openList.poll()
        val oldBestScore = bestScore

        if(node.item !in pSetInfos){
            val pInfo = createPInfo(X, node.item, trials, logPrior, jointPriorParams, alphaStrength, expertEv)
            pSetInfos[node.item] = pInfo
            if(pInfo.logProbability > Math.log(aliveThresh)  + bestScore){
                aliveList += node
                for(sibling in siblings(node)){
                    val pSetUnion = node.item + sibling.item
                    if(pSetUnion !in lattice){
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

    val normedReasonable = logNorm(aliveList.map { pSetInfos[it.item]!!})

    return normedReasonable
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
                     singleParentProb : Double, aliveThresh : Double) : ParentExtResult {
    val inReasonable = Math.log(0.99)
    val outReasonable = Math.log(0.01)
    val totalPsets = numAssignments(oldVocab + extraVocab)

    val priorMaps = rpLogProbs.mapValues { (_, rps) ->
        rps.flatMap { (oldPSet, oldLogProb) ->
            Sets.powerSet(extraVocab).map { extraSubset : Set<RandomVariable> ->
                val pSet = oldPSet + extraSubset
                val logProb = extraSubset.size * Math.log(singleParentProb) + (extraVocab.size - extraSubset.size) * Math.log(1 - singleParentProb) + oldLogProb
                Pair(pSet, logProb)
            }
        }.associate{ it }
    }

    val priorFuncs = priorMaps.mapValues{ (_, priorMap) ->
        { pSet : PSet -> inOutPrior(pSet, inReasonable, outReasonable, priorMap, totalPsets)}
    }


    val bestPSetScores = priorMaps.mapValues { (_, rps) -> rps.values.max() }

    val newReasonable = priorMaps.mapValues { (rv, rps) ->
        rps.filterValues { it >= bestPSetScores[rv]!! + Math.log(aliveThresh) }
            .map { it.key }
    }

    return ParentExtResult(priorFuncs, newReasonable)
}

fun inOutPrior(pSet : PSet, inLogProb : Double, outLogProb : Double, priorMap : Map<PSet, Double>, totalPSets : Int) =
    if(pSet in priorMap){
        inLogProb + priorMap[pSet]!!
    }
    else{
        outLogProb + Math.log(1.0 / (totalPSets - priorMap.size))
    }
