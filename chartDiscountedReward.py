# Import matplotlib etc.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import functools as fn
from os.path import basename, splitext


def plot_reward_results(df):
    print(df[1:100])
    plt.plot(df.index.get_values(), df["discounted_cum_reward"])
    plt.show()


def load_rewards_and_accumulate(file_path):
    df = pd.read_csv(file_path, index_col="time_stamp", names=["time_stamp", "reward"])
    full_timestamps = np.arange(0, 4000)

    df = df.reindex(full_timestamps, fill_value=0.0)
    df["discounted_cum_reward"] = discounted_accumulate(df["reward"], 0.99)
    return df


def load_accumulated_rewards(file_path):
    return pd.read_csv(file_path, index_col="time_stamp", names=["time_stamp", "reward", "discounted_cum_reward"])


def combine_reward_results(file_paths):
    dfs = [load_rewards_and_accumulate(fp) for fp in file_paths]
    avg_indices = ["reward", "discounted_cum_reward"]
    combined_df = fn.reduce(lambda df1, df2: df1[avg_indices] + df2[avg_indices], dfs)
    combined_df[avg_indices] = combined_df[avg_indices] / len(dfs)
    return combined_df


def merge_and_save(load_file_paths, save_file_path):
    combined_df = combine_reward_results(load_file_paths)
    combined_df.to_csv(save_file_path, header=False)


def compare_discounted_rewards(file_paths):
    plt.xlabel("Time Step")
    plt.ylabel("Discounted Reward")
    
    for file_path in file_paths:
        df = load_accumulated_rewards(file_path)
        df_label = splitext(basename(file_path))[0]
        plt.plot(df.index.get_values(), df["discounted_cum_reward"], label=df_label)
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
