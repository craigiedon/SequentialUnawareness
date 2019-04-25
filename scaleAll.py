import chartResults as ch
import os
from os.path import join
import sys
import re

# List dir from results folder
if len(sys.argv) != 4:
    print("Usage: python mergeResults.py <results_folder> <metric-name> <scale-factor>")
    sys.exit(1)

results_folder = sys.argv[1]
metric_name = sys.argv[2]

for experiment_folder in os.listdir(results_folder):
    if re.match(r'.*(true|random).*', experiment_folder):
        print experiment_folder

        abs_exp_folder = join(results_folder, experiment_folder)
        experiment_files = os.listdir(abs_exp_folder)

        policy_error_files = [join(abs_exp_folder, f)
                              for f in experiment_files
                              if re.match(r'.*-{0}.txt'.format(metric_name), f)]

        experiment_name = os.path.basename(experiment_folder)

        for policy_error_file in policy_error_files:
            ch.scale_and_save(policy_error_file, policy_error_file, 0.1)
