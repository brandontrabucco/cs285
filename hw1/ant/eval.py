"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.baselines.evaluate_policy import EvaluatePolicy
from gym.envs.mujoco.ant import AntEnv
import argparse


def run_experiment(policy_ckpt):

    mean, std = EvaluatePolicy(
        AntEnv,
        policy_ckpt=policy_ckpt,
        num_threads=10,
        max_path_length=1000,
        num_episodes=10).launch()
    print("mean: {} std: {}".format(mean, std))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_ckpt", type=str)
    args = parser.parse_args()
    run_experiment(args.policy_ckpt)
