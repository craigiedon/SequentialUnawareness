import chartResults as ch
import sys

merged_results_folder = sys.argv[1]
time_cutoff = int(sys.argv[2])

# ch.compare_vocab_size(["{0}/*Tolerance*VocabSize*".format(merged_results_folder), "{0}/*default*VocabSize*".format(merged_results_folder)], time_cutoff)
# ch.compare_actions_size(["{0}/*Tolerance*ActionsSize*".format(merged_results_folder), "{0}/*default*ActionsSize*".format(merged_results_folder)], time_cutoff)

metrics = [("discounted", "Cumulative Reward"), ("VocabSize", r'$|X^t| + |A^t|$')]
agent_groups = [["lowTolerance", "highTolerance", "default"], ["default", "non", "true", "random"]]
for (metric, label) in metrics:
    for agent_group in agent_groups:
        wild_paths = ["{0}/*{1}*{2}*".format(merged_results_folder, agent_type, metric) for agent_type in agent_group]
        ch.compare_results(wild_paths, "t", label, time_cutoff)

"""
ch.compare_results(["{0}/*default*discounted*".format(merged_results_folder),
                    "{0}/*non*discounted*".format(merged_results_folder),
                    "{0}/*true*discounted*".format(merged_results_folder),
                    "{0}/*random*discounted*".format(merged_results_folder)], time_cutoff)
"""
