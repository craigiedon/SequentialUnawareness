"""
- Load mpd itself
- Visualize the decision tree itself for each action
- Visualize just the RV nodes and their parents
- Visualize reward tree
"""
from graphviz import Digraph
import json
import sys
import uuid


def draw_decision_tree(action_name, dt_name, dt):
    dot = Digraph(comment="Decision Tree for: " + dt_name)
    dot.attr("graph", label="Decision tree: " + dt_name.upper() + "\nAction: " + action_name.upper(), labelloc="top")
    rec_show_decision_tree(dt, dot)
    return dot


def rec_show_decision_tree(dt_node, dot):
    node_id = str(uuid.uuid4())
    if isinstance(dt_node, list):
        dot.node(node_id, label=str(dt_node), shape="none")
    else:
        for decision_name, answer_map in dt_node.items():
            dot.node(node_id, label=decision_name.upper(), shape="oval")
            for answer_name, answer_node in answer_map.items():
                child_id = rec_show_decision_tree(answer_node, dot)
                dot.edge(node_id, child_id, label=answer_name)
    return node_id

    """
    for node_name, answer_map in dt_node.items():
        node_id = str(uuid.uuid4())
        dot.node(node_id, label=node_name.upper(), shape="oval")
        for answer_name, answer_node in answer_map.items():
            if isinstance(answer_node, list):
                leaf_id = str(uuid.uuid4())
                dot.node(leaf_id, label=str(answer_node), shape="none")
                dot.edge(node_id, leaf_id, label=answer_name)
            else:
                child_id = rec_show_decision_tree(answer_node, dot)
                dot.edge(node_id, child_id, label=answer_name)
        return node_id
    """


def draw_dependency_structure(action_name, dt_map):
    parent_sets = {}
    dot = Digraph(comment="DBN for action: " + action_name)
    dot.attr("graph", rankdir="LR", label="DBN for action: " + action_name, labelloc="top")
    print("DBN for action: " + action_name)
    for var_name, dt in dt_map.items():
        parent_sets[var_name] = p_set_from_dt(dt)
        for parent in parent_sets[var_name]:
            dot.edge(parent.upper(), var_name.upper() + '\'')

    print(parent_sets)
    return dot


def p_set_from_dt(dt):
    if(isinstance(dt, list)):
        return []

    mentioned_vars = list(dt.keys())
    for node_name, answer_map in dt.items():
        for answer_name, answer_node in answer_map.items():
            if not isinstance(answer_node, list):
                mentioned_vars.extend(p_set_from_dt(answer_node))
        return mentioned_vars


def draw_reward_tree(reward_tree):
    dot = Digraph(comment="Reward Tree")
    dot.attr("graph", label="Reward tree: ", labelloc="top")
    rec_show_decision_tree(reward_tree, dot)
    return dot


def load_mdp(file_path):
    json_file = open(file_path)
    mdp = json.load(json_file)
    json_file.close()
    return mdp


def show_all(mdp):
    action_maps = mdp["actions"]
    # Display individual decision trees
    for action_name, action_dts in action_maps.items():
        for dt_name, dt in action_dts.items():
            print(action_name, dt_name)
            drawing = draw_decision_tree(action_name, dt_name, dt)
            drawing.render("dt-" + action_name + "-" + dt_name, directory="mdpVisuals", view=False, cleanup=True)

    # Display DBN structure for each action
    for action_name, action_dts in action_maps.items():
        drawing = draw_dependency_structure(action_name, action_dts)
        drawing.render("depStruct-" + action_name, directory="mdpVisuals", view=True, cleanup=True)

    # Display Reward Tree
    drawing = draw_reward_tree(mdp["reward"])
    drawing.render("reward", directory="mdpVisuals", view=True, cleanup=True)


def main(file_path):
    load_mdp(file_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualizeMDPS.py <json_file_path>")

    main(sys.argv[1])
