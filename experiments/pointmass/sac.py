"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import multiprocessing
from cs285.baselines.off_policy.sac import SAC
from cs285.data.envs.pointmass_env import PointmassEnv


def run_experiment(experiment_id):
    SAC(
        PointmassEnv,
        logging_dir="./pointmass/sac/{}".format(experiment_id),
        hidden_size=256,
        num_hidden_layers=2,
        num_threads=10,
        max_path_length=10,
        max_num_steps=100000,
        reward_scale=1.0,
        discount=0.9,
        tau=0.05,
        policy_delay=1,
        initial_alpha=0.01,
        qf_learning_rate=0.0003,
        policy_learning_rate=0.0003,
        batch_size=32,
        num_epochs=10000,
        num_episodes_per_epoch=1,
        num_trains_per_epoch=10,
        num_episodes_before_train=20,
        num_epochs_per_eval=1,
        num_episodes_per_eval=10).launch()


if __name__ == "__main__":

    num_seeds = 5
    for seed in range(num_seeds):
        multiprocessing.Process(target=run_experiment, args=(seed,)).start()
