import chartResults as ch
import os
from os.path import join
import sys
import re

# List dir from results folder
if len(sys.argv) != 3:
    print("Usage: python mergeResults.py <results_folder> <metric-name>")
    sys.exit(1)

results_folder = sys.argv[1]
metric_name = sys.argv[2]

merge_folder = join(results_folder, os.pardir, "merged")
if not os.path.exists(merge_folder):
    os.mkdir(merge_folder)

for experiment_folder in os.listdir(results_folder):
    abs_exp_folder = join(results_folder, experiment_folder)
    experiment_files = os.listdir(abs_exp_folder)

    policy_error_files = [join(abs_exp_folder, f) for f in experiment_files if re.match(r'.*-{0}.txt'.format(metric_name), f)]

    experiment_name = os.path.basename(experiment_folder)
    print(experiment_name)
    ch.merge_and_save(policy_error_files, os.path.join(merge_folder, "{0}-{1}.csv".format(experiment_name, metric_name)))
