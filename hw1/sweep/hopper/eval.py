"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.core.eval_policy import eval_policy, eval_policy_variant
from gym.envs.mujoco.hopper import HopperEnv
import argparse
import matplotlib.pyplot as plt
import copy
import os


def run_experiment(data_dir):
    mean_returns = []
    for num_episodes in [2, 4, 6, 8, 10]:
        variant = copy.deepcopy(eval_policy_variant)

        variant["policy_ckpt"] = os.path.join(
            data_dir,
            "{}_episodes/policy.ckpt".format(num_episodes))

        variant["exploration_noise_std"] = 0.1
        variant["num_threads"] = 10
        variant["max_path_length"] = 1000
        variant["num_episodes"] = 10

        mean, std = eval_policy(variant, HopperEnv)
        mean_returns.append(mean)

    plt.clf()

    ax = plt.subplot(111)
    ax.plot(
        [2000, 4000, 6000, 8000, 10000],
        mean_returns)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xlabel("Number Of Demonstrations")
    ax.set_ylabel("Return Mean")

    ax.set_title("More Demonstrations Improves Performance: Hopper-v2")
    plt.savefig(os.path.join(
        data_dir, "sweep_mean_return.png"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    run_experiment(args.data_dir)
