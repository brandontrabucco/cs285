"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import multiprocessing
from cs285.baselines.behavior_cloning import behavior_cloning, behavior_cloning_variant
from gym.envs.mujoco.hopper import HopperEnv


def run_experiment(experiment_id):

    behavior_cloning_variant["logging_dir"] = "./hopper/behavior_cloning/{}".format(experiment_id)
    behavior_cloning_variant["hidden_size"] = 256
    behavior_cloning_variant["num_hidden_layers"] = 2
    behavior_cloning_variant["exploration_noise_std"] = 0.1
    behavior_cloning_variant["expert_policy_ckpt"] = "./hopper/expert_policy.ckpt"
    behavior_cloning_variant["num_threads"] = 10
    behavior_cloning_variant["max_path_length"] = 1000
    behavior_cloning_variant["max_num_steps"] = 1000000
    behavior_cloning_variant["batch_size"] = 256
    behavior_cloning_variant["num_epochs"] = 1000
    behavior_cloning_variant["num_episodes_per_epoch"] = 0
    behavior_cloning_variant["num_trains_per_epoch"] = 10
    behavior_cloning_variant["num_episodes_before_train"] = 10
    behavior_cloning_variant["num_epochs_per_eval"] = 1
    behavior_cloning_variant["num_episodes_per_eval"] = 10

    behavior_cloning(behavior_cloning_variant, HopperEnv)


if __name__ == "__main__":

    num_seeds = 5
    for seed in range(num_seeds):
        multiprocessing.Process(target=run_experiment, args=(seed,)).start()
