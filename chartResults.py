# Import matplotlib etc.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from glob import glob
import functools as fn
from os.path import basename, splitext


def plot_reward_results(df):
    plt.plot(df.index.get_values(), df["discounted_cum_reward"])
    plt.show()


def compare_plots(dfs):
    for df in dfs:
        plt.plot(df.index.get_values(), df["discounted_cum_reward"])
    plt.show()


def load_rewards_and_accumulate(file_path, time_steps):
    df = pd.read_csv(file_path, index_col="time_stamp", names=["time_stamp", "reward", "standard_error"])
    full_timestamps = np.arange(0, time_steps)

    df = df.reindex(full_timestamps, fill_value=0.0)
    df["discounted_cum_reward"] = discounted_accumulate(df["reward"], 0.99)
    return df


def load_time_stamped_val(file_path):
    df = pd.read_csv(file_path, index_col="time_stamp", names=["time_stamp", "value", "standard_error"])
    return df


def load_accumulated_rewards(file_path):
    return pd.read_csv(file_path, index_col="time_stamp", names=["time_stamp", "reward", "discounted_cum_reward"])


def combine_reward_results(wild_paths, time_steps):
    file_paths = [file_name for wild_path in wild_paths for file_name in glob(wild_path)]
    dfs = [load_rewards_and_accumulate(fp, time_steps) for fp in file_paths]
    avg_indices = ["reward", "discounted_cum_reward"]
    combined_df = fn.reduce(lambda df1, df2: df1[avg_indices] + df2[avg_indices], dfs)
    combined_df[avg_indices] = combined_df[avg_indices] / len(dfs)
    return combined_df


def merge_and_save(load_file_paths, save_file_path, time_steps):
    combined_df = combine_reward_results(load_file_paths, time_steps)
    combined_df.to_csv(save_file_path, header=False)


def compare_discounted_rewards(wild_paths):
    file_paths = [file_name for wild_path in wild_paths for file_name in glob(wild_path)]
    plt.xlabel("Time Step")
    plt.ylabel("Discounted Reward")

    for file_path in file_paths:
        df = load_rewards_and_accumulate(file_path, 10000)
        df_label = splitext(basename(file_path))[0].split("-")[-2]
        plt.plot(df.index.get_values(), df["discounted_cum_reward"], label=df_label)
    plt.legend(loc="best")
    plt.show()


def compare_vocab_size(wild_paths):
    file_paths = [file_name for wild_path in wild_paths for file_name in glob(wild_path)]
    plt.xlabel("Time Step")
    plt.ylabel("Vocab Size")

    for file_path in file_paths:
        df = load_time_stamped_val(file_path)
        df_label = splitext(basename(file_path))[0]
        plt.plot(df.index.get_values(), df["value"], label=df_label)
    plt.legend(loc="best")
    plt.show()


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
