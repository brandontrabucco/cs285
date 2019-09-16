"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.core.eval_policy import eval_policy, eval_policy_variant
from gym.envs.mujoco.ant import AntEnv
import argparse


def run_experiment(policy_ckpt):

    eval_policy_variant["exploration_noise_std"] = 0.1
    eval_policy_variant["policy_ckpt"] = policy_ckpt
    eval_policy_variant["num_threads"] = 10
    eval_policy_variant["max_path_length"] = 1000
    eval_policy_variant["num_episodes"] = 10

    mean, std = eval_policy(eval_policy_variant, AntEnv)
    print("mean: {} std: {}".format(mean, std))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_ckpt", type=str)
    args = parser.parse_args()
    run_experiment(args.policy_ckpts)
