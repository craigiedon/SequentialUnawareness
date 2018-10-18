import com.fasterxml.jackson.databind.JavaType
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import org.apache.commons.io.FilenameUtils
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.lang.IllegalArgumentException
import kotlin.reflect.KClass

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

fun <T : Any> loadJson(fileName : String, loadClass : KClass<T>) : T {
    if(!FilenameUtils.isExtension(fileName, "json")){
        throw IllegalArgumentException("$fileName does not have .json extension")
    }
    val mapper = jacksonObjectMapper().registerKotlinModule()
    return mapper.readValue(File(fileName), loadClass.java)
}

fun <T> loadJson(fileName : String, loadType : JavaType) : T {
    if(!FilenameUtils.isExtension(fileName, "json")){
        throw IllegalArgumentException("$fileName does not have .json extension")
    }
    val mapper = jacksonObjectMapper().registerKotlinModule()
    return mapper.readValue(File(fileName), loadType)
}

fun loadJsonTree(fileName : String) : JsonNode{
    if(!FilenameUtils.isExtension(fileName, "json")){
        throw IllegalArgumentException("$fileName does not have .json extension")
    }
    val mapper = jacksonObjectMapper().registerKotlinModule()
    val node = mapper.readTree(File(fileName))
    return node
}
