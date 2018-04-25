import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.readValue
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter

fun saveToJson(logData : Any, outputFolder : String, fileName: String) {
    saveToJson(logData, "$outputFolder/$fileName")
}

fun saveToJson(logData : Any, filePath : String) {
    val mapper = jacksonObjectMapper().registerKotlinModule()
    val jsonString = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(logData)

    val bufWriter = BufferedWriter(FileWriter("$filePath.json"))
    bufWriter.write(jsonString)
    bufWriter.close()
}

fun <T> loadJson(fileName : String, loadClass : Class<T>) : T {
    val mapper = jacksonObjectMapper().registerKotlinModule()
    return mapper.readValue(File(fileName), loadClass)
}

fun loadJsonTree(fileName : String) : JsonNode{
    val mapper = jacksonObjectMapper().registerKotlinModule()
    val node = mapper.readTree(File(fileName))
    return node
}
