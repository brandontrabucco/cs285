"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import multiprocessing
from cs285.baselines.off_policy.sac import SAC
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


def run_experiment(experiment_id):
    SAC(
        HalfCheetahEnv,
        logging_dir="./half_cheetah/sac/{}".format(experiment_id),
        hidden_size=256,
        num_hidden_layers=2,
        num_threads=10,
        max_path_length=1000,
        max_num_steps=1000000,
        reward_scale=1.0,
        discount=0.99,
        tau=0.005,
        policy_delay=1,
        initial_alpha=0.01,
        qf_learning_rate=0.0003,
        policy_learning_rate=0.0003,
        batch_size=256,
        num_epochs=10000,
        num_episodes_per_epoch=1,
        num_trains_per_epoch=1000,
        num_episodes_before_train=10,
        num_epochs_per_eval=10,
        num_episodes_per_eval=10).launch()


if __name__ == "__main__":

    num_seeds = 5
    for seed in range(num_seeds):
        multiprocessing.Process(target=run_experiment, args=(seed,)).start()
