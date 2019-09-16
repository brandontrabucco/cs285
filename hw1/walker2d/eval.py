"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.core.eval_policy import eval_policy, eval_policy_variant
from gym.envs.mujoco.walker2d import Walker2dEnv
import argparse
import numpy as np


def run_experiment(policy_ckpts):

    mean_returns = []
    for policy_ckpt in policy_ckpts:

        eval_policy_variant["exploration_noise_std"] = 0.1
        eval_policy_variant["policy_ckpt"] = policy_ckpt
        eval_policy_variant["num_threads"] = 10
        eval_policy_variant["max_path_length"] = 1000
        eval_policy_variant["num_episodes"] = 10

        paths, mean_return, steps = eval_policy(eval_policy_variant, Walker2dEnv)
        mean_returns.append(mean_return)

    print("mean: {}    std: {}".format(
        np.mean(mean_returns), np.std(mean_returns)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_ckpts", type=str, nargs="+")
    args = parser.parse_args()
    run_experiment(args.policy_ckpts)
