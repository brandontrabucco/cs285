"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import multiprocessing
from cs285.baselines.dagger import dagger, dagger_variant
from gym.envs.mujoco.hopper import HopperEnv


def run_experiment(experiment_id):

    dagger_variant["logging_dir"] = "./hopper/dagger/{}".format(experiment_id)
    dagger_variant["hidden_size"] = 256
    dagger_variant["num_hidden_layers"] = 2
    dagger_variant["exploration_noise_std"] = 0.1
    dagger_variant["expert_policy_ckpt"] = "./hopper/expert_policy.ckpt"
    dagger_variant["num_threads"] = 10
    dagger_variant["max_path_length"] = 1000
    dagger_variant["max_num_steps"] = 1000000
    dagger_variant["batch_size"] = 256
    dagger_variant["num_epochs"] = 1000
    dagger_variant["num_episodes_per_epoch"] = 1
    dagger_variant["num_trains_per_epoch"] = 10
    dagger_variant["num_episodes_before_train"] = 10
    dagger_variant["num_epochs_per_eval"] = 1
    dagger_variant["num_episodes_per_eval"] = 10

    dagger(dagger_variant, HopperEnv)


if __name__ == "__main__":

    num_seeds = 5
    for seed in range(num_seeds):
        multiprocessing.Process(target=run_experiment, args=(seed,)).start()
