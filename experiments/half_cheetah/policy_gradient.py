"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import multiprocessing
from cs285.baselines.policy_gradient import policy_gradient, policy_gradient_variant
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


def run_experiment(experiment_id):

    policy_gradient_variant["logging_dir"] = "./half_cheetah/policy_gradient/{}".format(experiment_id)
    policy_gradient_variant["hidden_size"] = 256
    policy_gradient_variant["num_hidden_layers"] = 2
    policy_gradient_variant["num_threads"] = 10
    policy_gradient_variant["max_path_length"] = 150
    policy_gradient_variant["reward_scale"] = 1.0
    policy_gradient_variant["discount"] = 0.99
    policy_gradient_variant["policy_learning_rate"] = 0.02
    policy_gradient_variant["batch_size"] = 333
    policy_gradient_variant["num_epochs"] = 10000
    policy_gradient_variant["num_episodes_per_epoch"] = 333
    policy_gradient_variant["num_trains_per_epoch"] = 1
    policy_gradient_variant["num_episodes_before_train"] = 0
    policy_gradient_variant["num_epochs_per_eval"] = 1
    policy_gradient_variant["num_episodes_per_eval"] = 20

    policy_gradient(policy_gradient_variant, HalfCheetahEnv)


if __name__ == "__main__":

    num_seeds = 5
    for seed in range(num_seeds):
        multiprocessing.Process(target=run_experiment, args=(seed,)).start()
