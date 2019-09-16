"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import argparse
from cs285.core.eval_policy import eval_policy, eval_policy_variant
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


def run_experiment(policy_ckpt):

    eval_policy_variant["exploration_noise_std"] = 0.1
    eval_policy_variant["policy_ckpt"] = policy_ckpt
    eval_policy_variant["num_threads"] = 10
    eval_policy_variant["max_path_length"] = 1000
    eval_policy_variant["num_episodes"] = 10

    paths, mean_return, steps = eval_policy(eval_policy_variant, HalfCheetahEnv)
    print("{}".format(mean_return))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_ckpt", type=str)
    args = parser.parse_args()
    run_experiment(args.policy_ckpt)
