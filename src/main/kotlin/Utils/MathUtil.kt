package Utils

fun pow(a : Int, b : Int) = Math.pow(a.toDouble(), b.toDouble()).toInt()

fun logOfBase(num : Double, base: Double) : Double{
    return Math.log(num) / Math.log(base)
}

fun lnToDifferentBase(lnResult : Double, base : Double) : Double{
    return lnResult / Math.log(base)
}

fun sumInnerLnProbs(lnProbs : List<Double>) : Double {
    val maxLog = lnProbs.max()!!
    val shiftedLogs = lnProbs.map { it - maxLog }
    val exponentiatedVals = shiftedLogs.map { Math.exp(it) }
    return Math.log(exponentiatedVals.sum()) + maxLog
}

fun logNormalize(unNormalizedLogs : List<Double>) : List<Double>{
    val logTotal = sumInnerLnProbs(unNormalizedLogs)
    return unNormalizedLogs.map { it - logTotal }
}

fun <T> logNormalize(unNormalizedMap : Map<T, Double>) : Map<T, Double>{
    val logTotal = sumInnerLnProbs(unNormalizedMap.values.toList())
    return unNormalizedMap.mapValues { it.value - logTotal }
}

fun currentTimeSecs() = System.currentTimeMillis() / 1000.0
