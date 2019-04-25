# Import matplotlib etc.
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import pandas as pd
import sys
from glob import glob
from math import sqrt
import functools as fn
from os.path import basename, splitext


def plot_reward_results(df):
    plt.plot(df.index.get_values(), df["discounted_cum_reward"])
    plt.show()


def compare_plots(dfs):
    for df in dfs:
        plt.plot(df.index.get_values(), df["discounted_cum_reward"])
    plt.show()


"""
def load_accumulated_rewards(file_path):
    df = pd.read_csv(file_path, names=["time_stamp", "reward", "discounted_cum_reward", "sem"])
    # df["discounted_cum_reward"] = discounted_accumulate(df["value"], 0.99)
    return df
"""


def load_time_stamped_val(file_path):
    df = pd.read_csv(file_path, names=["time_stamp", "value", "sem"])
    return df


def combine_results(wild_paths):
    file_paths = [file_name for wild_path in wild_paths for file_name in glob(wild_path)]
    dfs = [pd.read_csv(fp, names=["time_stamp", "value"]) for fp in file_paths]

    combined_df = pd.concat(dfs).groupby("time_stamp")

    avg_df = combined_df.mean()
    avg_df["sem"] = combined_df.sem()["value"]

    return avg_df


def merge_and_save(load_file_paths, save_file_path):
    combined_df = combine_results(load_file_paths)
    combined_df.to_csv(save_file_path, header=False)


def scale_and_save(load_file_path, save_file_path, scale_factor):
    df = pd.read_csv(load_file_path, names=["time_stamp", "value"])
    df["value"] = df["value"] * scale_factor
    df.to_csv(save_file_path, header=False)


def discounted_accumulate_and_save(load_file_path, save_file_path, discount_factor):
    df = pd.read_csv(load_file_path, names=["time_stamp", "value"])
    df["value"] = discounted_accumulate(df["value"], discount_factor)
    df.to_csv(save_file_path, header=False)


def compare_discounted_rewards(wild_paths, time_cutoff=None):
    file_paths = [file_name for wild_path in wild_paths for file_name in glob(wild_path)]
    plt.rcParams["figure.figsize"] = (4.0 * 0.9, 3.0 * 0.9)
    plt.xlabel("t")
    plt.ylabel("Cumulative Reward")
    plt.ticklabel_format(axis="x", style='sci', scilimits=(1, 3))

    #linestyle_cycler = cycle([(None, None), (1, 2), (5, 2), (2, 5)])
    marker_cycler = cycle(['^', 'o', 's', 'x'])
    print(file_paths)

    for file_path in file_paths:
        df = load_time_stamped_val(file_path)
        df_label = splitext(basename(file_path))[0].split("-")[-2]
        if df_label == "nonConservative":
            df_label = "nonCon"

        if time_cutoff is not None:
            time_steps = df["time_stamp"].iloc[0:time_cutoff]
            disc_rewards = df["value"].iloc[0:time_cutoff]
        else:
            time_steps = df["time_stamp"]
            disc_rewards = df["value"]

        everyN = 50
        plt.plot(time_steps[0::everyN], disc_rewards[0::everyN], label=df_label, marker=next(marker_cycler), markevery=2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def compare_results(wild_paths, x_label, y_label, time_cutoff=None):
    file_paths = [file_name for wild_path in wild_paths for file_name in glob(wild_path)]

    plt.rcParams["figure.figsize"] = (4.0 * 0.9, 3.0 * 0.9)
    marker_cycler = cycle(['^', 'o', 's', 'x'])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ticklabel_format(axis="x", style='sci', scilimits=(1, 3))

    #linestyle_cycler = cycle([(None, None), (1, 2), (5, 2), (2, 5)])

    for file_path in file_paths:
        df = load_time_stamped_val(file_path)
        df_label = splitext(basename(file_path))[0].split("-")[-2]
        print df["value"]
        if time_cutoff is not None:
            time_steps = df.index.get_values()[0:time_cutoff]
            values = df["value"][0:time_cutoff]
            sems = df["sem"][0:time_cutoff]
        else:
            time_steps = df.index.get_values()
            values = df["value"]
            sems = df["sem"][0:time_cutoff]

        everyN = 250
        plt.plot(time_steps[0::everyN], values[0::everyN], label=df_label, marker=next(marker_cycler), markevery=2, linewidth=0.9)
        plt.fill_between(time_steps[0::everyN], values[0::everyN] + sems[0::everyN], values[0::everyN] - sems[0::everyN], alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def compare_vocab_size(wild_paths, time_cutoff=None):
    compare_results(wild_paths, "t", r'$|X^t|$', time_cutoff)


def compare_actions_size(wild_paths, time_cutoff=None):
    compare_results(wild_paths, "t", r'$|A^t|$', time_cutoff)


def discounted_accumulate(nums, discount):
    prev_sum = 0
    discounted_cum_sums = []
    for num in nums:
        new_sum = prev_sum * discount + num
        discounted_cum_sums.append(new_sum)
        prev_sum = new_sum
    return discounted_cum_sums


if __name__ == "__main__":
    compare_discounted_rewards(sys.argv[1:])
