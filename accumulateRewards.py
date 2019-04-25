import chartResults as ch
import os
from os.path import join
import sys
import re

# List dir from results folder
if len(sys.argv) != 2:
    print("Usage: python mergeResults.py <results_folder>")
    sys.exit(1)

results_folder = sys.argv[1]

for experiment_folder in os.listdir(results_folder):
    abs_exp_folder = join(results_folder, experiment_folder)
    experiment_files = os.listdir(abs_exp_folder)

    load_files = [join(abs_exp_folder, f) for f in experiment_files if re.match(r'.*-reward.txt', f)]

    # TODO: instead of saving to same place, save to a separate file for discounted accumulated reward
    for load_file in load_files:
        save_file = load_file.replace("reward", "discountedCumReward")
        print(load_file)
        print(save_file)
        ch.discounted_accumulate_and_save(load_file, save_file, 0.99)
