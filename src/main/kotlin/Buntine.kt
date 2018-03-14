import Utils.defaultMap
import Utils.logNormalize
import Utils.sumInnerLnProbs
import java.util.*

/* So, what do I actually want to do in here?
 - construction of dbns: given reasonable parent parts per child, and per action, form the appropriate DBNs for each
 - consistency checking afterwards: Do all vars have some route to the reward (if not, what will you do? Ask alex about the details of this when you get there)
 */

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
)


class BuntineUpdater(private val aliveThresh: Double, private val pseudoCountSize : Double, private val singleParentProb : Double) {

    fun addBeliefVariable(newVar: RandomVariable, timeStep: Int, seqTrials: List<SequentialTrial>, expertEv: List<DirEdge>, dbnInfo: DBNInfo): DBNInfo {
        val oldVocab = dbnInfo.bestPInfos.keys
        val oldVocabReasonablePs = dbnInfo.reasonableParents.mapValues { (rv, _) ->
            structuralUpdate(rv, dbnInfo.pSetPriors[rv]!!, oldVocab, seqTrials, expertEv, dbnInfo.priorJointParams[rv]!!, aliveThresh, pseudoCountSize)
        }
        val oldVocabBest = bestParents(oldVocabReasonablePs)
        val oldVocabCPTs = dbnInfo.cptsITI.mapValues { (rv, cpt) -> changeVocab(cpt, oldVocabBest[rv]!!.parentSet) }

        // Also, remake your dbn at this point (decision trees and all):
        val priorJointParams = oldVocabCPTs.mapValues { (rv, cpt) -> convertToJointProbTree(cpt, dbnInfo.priorJointParams[rv]!!, pseudoCountSize) }
        val (logPriors, reasonableParents) = posteriorToPrior(oldVocabReasonablePs, oldVocab, setOf(newVar), priorJointParams, singleParentProb, aliveThresh, pseudoCountSize)


        val bestParents = bestParents(reasonableParents)
        val cptsITI = initialCPTsITI(oldVocab + newVar, bestParents, priorJointParams, pseudoCountSize)
        val dbn = oldVocabCPTs.mapValues { (rv, cpt) -> convertToCPT(cpt, oldVocabBest[rv]!!.priorParams, pseudoCountSize) }

        return DBNInfo(reasonableParents, bestParents, cptsITI, priorJointParams, dbn, logPriors, timeStep)
        // Throw away old trial counts
        //agentTrialHist[a]!!.clear()
    }

    fun initialDBNInfo(actions: Set<Action>, vocab: Set<RandomVariable>, expertEv: List<DirEdge>): Map<Action, DBNInfo> {
        return actions.associate { a ->
            val priorJointParams = unifStartIDTransJointDBN(vocab, 0.1)
            val pSetPriors = initialPSetPriors(vocab, singleParentProb)
            val reasonableParents = structuralUpdate(vocab, emptyList(), expertEv, pSetPriors, aliveThresh, priorJointParams, pseudoCountSize)
            val bestPInfos = bestParents(reasonableParents)
            val cptsITI = initialCPTsITI(vocab, bestPInfos, priorJointParams, pseudoCountSize)
            val dbn = cptsITIToDBN(bestPInfos, cptsITI, pseudoCountSize)

            val dbnInfo = DBNInfo(reasonableParents, bestPInfos, cptsITI, priorJointParams, dbn, pSetPriors, 0)
            Pair(a, dbnInfo)
        }
    }
}


fun minParentsPrior(vocab : Set<RandomVariable>, extraParentCost : Double) : LogPrior<PSet>{
    val singleParentProbabilities = vocab.associate { Pair(it, extraParentCost) }
    return { pSet -> vocab.sumByDouble { rv ->
        val spCost = Math.log(singleParentProbabilities[rv]!!)
        if(rv in pSet) spCost else 1 - spCost
    }}
}

fun parameterUpdate(child : RandomVariable,
                    reasonableParents :  List<SeqPInfo>,
                    seqTrial : SequentialTrial,
                    expertEvidence : List<DirEdge>,
                    alphaTotal: Double) : List<SeqPInfo>{
    val newReasonableParentSets = reasonableParents.map { pInfo ->
        //Sample counts are updated incrementally here
        pInfo.addTrial(seqTrial)

        val parentAssgn = seqTrial.prevState.filterKeys { it in pInfo.parentSet }
        val childVal = seqTrial.currentState[child]!!

        val countJoint = pInfo.count(childVal, parentAssgn)
        val countMarginal = pInfo.count(parentAssgn)

        val alphaJoint = alphaTotal * pInfo.jointPrior(childVal, parentAssgn)
        val alphaMarginal = alphaTotal * pInfo.marginalPrior(parentAssgn)

        //Update Structural Probability
        val newLogProb = pInfo.logProbability + Math.log(countJoint + alphaJoint - 1) - Math.log(countMarginal + alphaMarginal - 1)
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

//- structural update: Given a vocab, stateHist, a prior, a child, a parent vocab and a threshold value, output a set of reasonable parents
data class LatticeNode<V> constructor(val item : V, val parents : MutableList<LatticeNode<V>> = ArrayList()){
    val children = mutableListOf<LatticeNode<V>>()
    init {
        parents.forEach { it.children.add(this) }
    }
}

typealias PSetNode = LatticeNode<PSet>
typealias LogPrior<T> = (T) -> Double

fun createPInfo(child : RandomVariable, pSet: PSet, trials : List<SequentialTrial>, logPrior: LogPrior<PSet>, priorJointParams: DecisionTree<Factor>, priorSampleSize : Double, expertEv : List<DirEdge>) : SeqPInfo{
    val counts = trials
        .groupBy { it.prevState.filterKeys { it in pSet } }
        .mapValues { (_, trials) -> trials
            .groupBy { it.currentState[child]!!}
            .mapValues{ it.value.count() }
            .toOrderedList(child)
            .toMutableList()
        }

    val priorParams = allAssignments(pSet.toList()).associate { pAssign ->
        Pair(pAssign, jointQuery(pAssign, priorJointParams))
    }

    val score = if(violatesEvidence(child, pSet, expertEv)){
        Math.log(0.0)
    }
    else{
        logPrior(pSet) + BDeuScore(child, pSet, counts, priorParams, priorSampleSize)
    }
    return SeqPInfo(child, pSet, score, counts, priorParams)
}

fun <V> siblings(node : LatticeNode<V>) : List<LatticeNode<V>> =
    node.parents.flatMap { p -> p.children.filter { it.item != node.item } }

fun structuralUpdate(vocab: Set<RandomVariable>, trials: List<SequentialTrial>, expertEv: List<DirEdge>, priors: Map<RandomVariable, LogPrior<PSet>>, aliveThresh: Double, priorDBN: DynamicBayesNet, alphaStrength: Double): Map<RandomVariable, List<SeqPInfo>> {
    return priors.mapValues { (rv, prior) ->
        structuralUpdate(rv, prior, vocab, trials, expertEv, priorDBN[rv]!!, aliveThresh, alphaStrength)
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

            if(oldBestScore <= bestScore){
                aliveList = revisedAliveList(aliveThresh, aliveList, pSetInfos)
            }
        }
    }

    val normedReasonable = logNorm(aliveList.map { pSetInfos[it.item]!!})

    return normedReasonable
}

fun revisedAliveList(aliveThresh: Double, aliveList: List<PSetNode>, pSetInfos: Map<PSet, SeqPInfo>) : List<PSetNode> {
    val bestScore = pSetInfos.values.map { it.logProbability }.max()!!
    return aliveList.filter { isAlive(pSetInfos[it.item]!!.logProbability, aliveThresh, bestScore) }
}

fun isAlive(logProbability: Double, aliveThresh: Double, logBest: Double) =
    logProbability > Math.log(aliveThresh) + logBest

fun bestParents(parentChoices : Map<RandomVariable, Map<PSet, Double>>) : DBNStruct =
    parentChoices.mapValues{ (_, reasonableParents) -> reasonableParents.maxBy { it.value }!!.key}

@JvmName("BestParentsPInfo")
fun bestParents(parentChoices : Map<RandomVariable, List<SeqPInfo>>) : Map<RandomVariable, SeqPInfo> =
    parentChoices.mapValues{ (_, reasonableParents) -> reasonableParents.maxBy { it.logProbability }!!}

// Variables which are not in reward domain (or are not the ancestor of some rv in reward domain) are useless
fun uselessVariables(vocab : Set<RandomVariable>, dbnStructs : List<DBNStruct>, rewardScope : Set<RandomVariable>) : List<RandomVariable> {
    val descendantsPerDBN = dbnStructs.map { descendants(vocab, it) }
    val nonRewardScope = vocab.filter { it !in rewardScope }
    return nonRewardScope
        .filter { rv ->
            !descendantsPerDBN.any{ descendants ->
                descendants[rv]!!.any{ it in rewardScope}
            }
        }
}

fun <T> parentToChildMap(parentMap : Map<T, Set<T>>) : Map<T, Set<T>>{
    val vocab = parentMap.keys
    return parentMap.mapValues { (parent, _) ->
        vocab.filter { child -> parent in parentMap[child]!! }.toSet()
    }
}

fun descendants(vocab : Set<RandomVariable>, dbnStruct : DBNStruct) : Map<RandomVariable, Set<RandomVariable>>{
    val childMap = parentToChildMap(dbnStruct)
    val descendants = HashMap(childMap)

    var changed = false
    while(!changed){
        for(rv in vocab){
            val oldDescendants = descendants[rv]!!
            val nextStep = oldDescendants.flatMap{ descendants[it]!! }.toSet()
            val newDescendants = oldDescendants + nextStep
            if (oldDescendants != newDescendants){
                changed = true
                descendants[rv] = newDescendants
            }
        }
    }
    return descendants
}
