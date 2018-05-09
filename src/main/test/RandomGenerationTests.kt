import org.junit.Test

class RandomGenerationTests{
    @Test
    fun generateInitialState_generatesAtLeastOneOfEachTypeGlueCombination(){
        val mdp = loadMDP("mdps/medium-factory.json")

        val startStates = ArrayList<RVAssignment>()
        for(i in 1..10000){
            startStates.add(generateInitialState(mdp.vocab.toList(), mdp.startStateDescriptions, mdp.terminalDescriptions))
        }

        val combos = startStates.groupBy { it.filterKeys { it.name == "glue"  || it.name == "bolts"} }
        println("Done")
    }
}