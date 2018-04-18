package Utils

import java.util.*

fun <T> powerSet(items: List<T>, minSize : Int, maxSize : Int) : List<List<T>>{
    fun powerSetRec(subItems: List<T>) : List<List<T>> {
        if(subItems.isEmpty()) return listOf(emptyList())

        val subPowerSet = powerSetRec(subItems.subList(1, subItems.size))
        return (subPowerSet + subPowerSet.map { it + subItems[0] }).filter { it.size <= maxSize }
    }

    return powerSetRec(items).filter { it.size >= minSize }
}

fun <K,V> isSubset(subset : Map<K,V>, superset : Map<K,V>) =
    superset.keys.size >= subset.keys.size && subset.all { superset[it.key] == it.value }

fun <T : Number> repeatNum(item : T, times : Int) : List<T>{
    return (1..times).map { item }
}

fun <T> orderedPairs(l1 : Collection<T>) =
    l1.flatMap { item1 -> l1
        .filter { item2 -> item2 != item1 }
        .map { item2 -> Pair(item1, item2) } }

fun <T> repeatList(l : List<T>, times : Int) : List<T>{
    val repeatedList = ArrayList<T>()
    for(i in 1..times){
        repeatedList.addAll(l)
    }
    return repeatedList
}

fun <K, V> project(original: Map<K,V>, projectionVocab: Collection<K>) =
    original.filterKeys { projectionVocab.contains(it) }

fun <T> plusIf(l : List<T>, item : T, condition : Boolean) = if(condition) l + item else l

fun <T> findPairFromSeparateGroups(groups : List<List<T>>, predicate : (T, T) -> Boolean) : Pair<T, T>? {
    if(groups.size < 2)
        return null

    for(group in groups){
        val otherGroupItems = groups.filter { it != group }.flatten()
        val allPairs = cartesianProduct(group, otherGroupItems)
        val match = allPairs.find { predicate(it.first, it.second) }
        if(match != null){
            return match
        }
    }
    return null
}

fun <T> cartesianProduct(l1 : List<T>, l2 : List<T>) : Sequence<Pair<T, T>>{
    if(l1.isEmpty() || l2.isEmpty())
        throw IllegalArgumentException("Cannot have cartesian product for empty lists")
    return l1.asSequence()
        .flatMap {
            item1 -> l2.asSequence().map {
                item2 -> Pair(item1, item2)
            }
        }
}

fun <X> List<X>.swapped(firstIndex: Int, secondIndex: Int): List<X> {
    val reorderedList = ArrayList(this)
    reorderedList[firstIndex] = this[secondIndex]
    reorderedList[secondIndex] = this[firstIndex]
    return reorderedList
}

fun <T> Pair<T, T>.contains(item : T) = this.first == item || this.second == item

fun Collection<Int>.product() = this.fold(1, { a, b -> a * b })
fun <T> Collection<T>.productByDouble(converter: (T) -> Double) = this.fold(1.0, { a, b -> a * converter(b) })

fun <T> Collection<T>.productBy(converter : (T) -> Int) = this.map{converter(it)}.product()

fun <T, S> Pair<T,T>.map(f : (T) -> S) = Pair(f(this.first), f(this.second))

fun <K,V> deleteRandomValue(dict : Map<K,V>) : Map<K,V>{
    val keyList = dict.keys.toList()
    val randomIndex = (Math.random() * dict.keys.size).toInt()
    return dict.filterKeys { it != keyList[randomIndex] }
}

fun <K, V> defaultMap(keys : Collection<K>, defaultVal : V) =
    keys.associate { key -> Pair(key, defaultVal) }.toMutableMap()

fun <K, V> merge(m1 : Map<K, V>, m2 : Map<K,V>, mergeFunc: (V, V) -> V) : MutableMap<K,V>{
    val newMap = HashMap(m1)
    for((key, value) in m2){
        if(key in newMap){
            newMap[key] = mergeFunc(newMap[key]!!, value)
        }
        else{
            newMap[key] = value
        }
    }
    return newMap
}

fun zeros(n : Int) : MutableList<Int>{
    return (1..n).map { 0 }.toMutableList()
}

fun <T> allEqual(l1 : List<T>) : Boolean{
    if(l1.isEmpty()){
        throw IllegalArgumentException("Must have at least one item in list to test for equality")
    }
    return l1.all { l1.first() == it }
}

fun <T> concat(l1 : List<T>, l2: List<T>) = l1 + l2

fun <V> List<V>.everyNth(n : Int) : List<V> =
    this.filterIndexed { i, _ -> (i + 1) % n == 0}

sealed class Either<out V, out E>
data class Result<out T>(val result : T) : Either<T, Nothing>()
data class Error<out T> (val error : T) : Either<Nothing , T>()
