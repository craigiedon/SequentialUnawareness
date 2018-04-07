/*
    Convert from iti: leaf single example - return leaf with reward
    Convert from iti: leaf multi example - return leaf with average reward
    Convert from iti: decision - return DT with corresponding factors

    Convert to CPT: leaf - leaf with corresponding factor (including prior)
    Convert to CPT: decision tree - corresponding decision tree

    Joint query: Fully specified p assignment: Match leaf and return
    Joint query: Empty p assignment : add all leaves
    Joint query: Partial p assignment (one parent on 2 parent tree): add specific leaves

    Convert to joint prob tree : Leaf -> minimal conversion, counts can just be scaled
    Convert to joint prob tree : PDecision -> scale by total counts in whole tree

    Incremental SVI: Single reward step from boutillier: Takes one step
    Value tree making redundant distinction: prune first then add on new

    Apply expert advice: advice matches exactly: Change leaf
    advice is over specific: alter single matched leaf
    advice is under-specified : throw exception

    Prune: Leaf - return same leaf
    DT Decision: Single step, range is far apart: Return original
    DT Decision: Single step, range is close: Return leaf
    DT Decision: Recursive decsion: entire thing within range: return leaf
    DT Decision: Recursive decision: One step collapsable, but total range above: only do single collapse

    Change allowed vocab stuff (should probably go into ITITests)
 */