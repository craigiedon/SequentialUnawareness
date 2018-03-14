import Utils.repeatList
import Utils.repeatNum

fun uniformJoint(scope : List<RandomVariable>) =
    repeatNum(1.0 / numAssignments(scope), numAssignments(scope))

fun uniformCPT(child : RandomVariable, parents : List<RandomVariable>) =
    repeatList(repeatNum(1.0 / child.domainSize, child.domainSize), numAssignments(parents))
