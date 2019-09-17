"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import multiprocessing
from cs285.baselines.sac import sac, sac_variant
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


def run_experiment(experiment_id):

    sac_variant["logging_dir"] = "./half_cheetah/sac/{}".format(experiment_id)
    sac_variant["hidden_size"] = 256
    sac_variant["num_hidden_layers"] = 2
    sac_variant["num_threads"] = 10
    sac_variant["max_path_length"] = 1000
    sac_variant["max_num_steps"] = 1000000
    sac_variant["reward_scale"] = 1.0
    sac_variant["discount"] = 0.99
    sac_variant["tau"] = 0.005
    sac_variant["policy_delay"] = 1
    sac_variant["initial_alpha"] = 0.01
    sac_variant["qf_learning_rate"] = 0.0003
    sac_variant["policy_learning_rate"] = 0.0003
    sac_variant["batch_size"] = 256
    sac_variant["num_epochs"] = 10000
    sac_variant["num_episodes_per_epoch"] = 1
    sac_variant["num_trains_per_epoch"] = 1000
    sac_variant["num_episodes_before_train"] = 10
    sac_variant["num_epochs_per_eval"] = 10
    sac_variant["num_episodes_per_eval"] = 10

    sac(sac_variant, HalfCheetahEnv)


if __name__ == "__main__":

    num_seeds = 5
    for seed in range(num_seeds):
        multiprocessing.Process(target=run_experiment, args=(seed,)).start()
