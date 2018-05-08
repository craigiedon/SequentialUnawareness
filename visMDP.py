from graphviz import Digraph
import json
import uuid
import sys
import os
import re


def rec_show_dt(dt_node, dot):
    node_id = str(uuid.uuid4())
    if "passBranch" in dt_node and "failBranch" in dt_node:
        rv_name = dt_node["rvTest"]["first"]["name"].upper()
        print(dt_node["rvTest"])
        test_val = dt_node["rvTest"]["first"]["domain"][dt_node["rvTest"]["second"]]
        dot.node(node_id, label=rv_name.upper() + "=" + test_val, shape="oval")
        pass_child = dt_node["passBranch"]
        pass_id = rec_show_dt(pass_child, dot)
        dot.edge(node_id, pass_id, label="t")

        fail_child = dt_node["failBranch"]
        fail_id = rec_show_dt(fail_child, dot)
        dot.edge(node_id, fail_id, label="f")
    else:
        dot.node(node_id, label=str(dt_node["value"]), shape="none")
    return node_id


def show_dependency_structure(problem_name, action_name, dep_struct):
    parent_sets = {}
    dot = Digraph(comment="DBN for action: " + action_name)
    dot.attr("graph", rankdir="LR", label="DBN for action: " + action_name,
             labelloc="top")
    print("DBN for action: " + action_name)
    for child_name, p_set in dep_struct.items():
        for parent in p_set:
            dot.edge(parent.upper(), child_name.upper() + '\'')

    print(parent_sets)
    dot.render(filename=problem_name + "-" + action_name,
               view=True,
               cleanup=True)


def show_cpt(action_name, dt_name, dt):
    print("CPT for ", dt_name)
    dot = Digraph(comment="CPT Tree for: " + dt_name)
    dot.attr("graph", label="CPT tree: " + dt_name.upper() + "\nAction: " + action_name.upper(), labelloc="top")
    rec_show_dt(dt, dot)
    dot.view()


def show_agent_reward_tree(reward_tree):
    dot = Digraph(comment="Reward Tree")
    dot.attr("graph", label="Reward tree: ", labelloc="top")
    rec_show_dt(reward_tree, dot)
    dot.view()
    print(reward_tree)


def show_policy_tree(problem_name, policy_tree):
    dot = Digraph(comment="Policy Tree")
    dot.attr("graph", label="Policy Tree: ", labelloc="top")
    rec_show_dt(policy_tree, dot)
    dot.render(filename=problem_name + "-policy",
               view=True,
               cleanup=True)


def show_all_dep_structures(structs_folder, test_name_prefix):
    file_matcher = test_name_prefix + r'-transition-(.*).json$'
    dep_struct_files = [x for x in os.listdir(structs_folder)
                        if re.match(file_matcher, x)]
    print("Matching files: ", dep_struct_files)
    for f_name in dep_struct_files:
        action_name = re.match(file_matcher, f_name).group(1)
        dep_struct = load_json(structs_folder + "/" + f_name)
        show_dependency_structure(test_name_prefix, action_name, dep_struct)


def load_json(file_path):
    j_file = open(file_path)
    loaded_json = json.load(j_file)
    j_file.close()
    return loaded_json


def show_policy_from_file(file_path):
    policy_tree = load_json(file_path)
    show_policy_tree(os.path.splitext(file_path)[0], policy_tree)
