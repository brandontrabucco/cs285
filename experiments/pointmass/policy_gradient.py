"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import multiprocessing
from cs285.baselines.on_policy.policy_gradient import PolicyGradient
from cs285.data.envs.pointmass_env import PointmassEnv


def run_experiment(experiment_id):
    PolicyGradient(
        PointmassEnv,
        logging_dir="./pointmass/policy_gradient/{}".format(experiment_id),
        hidden_size=256,
        num_hidden_layers=2,
        max_num_paths=30,
        num_threads=10,
        max_path_length=10,
        reward_scale=1.0,
        discount=0.99,
        policy_learning_rate=0.02,
        batch_size=30,
        num_epochs=10000,
        num_episodes_per_epoch=30,
        num_trains_per_epoch=1,
        num_episodes_before_train=0,
        num_epochs_per_eval=1,
        num_episodes_per_eval=30).launch()


if __name__ == "__main__":

    num_seeds = 5
    for seed in range(num_seeds):
        multiprocessing.Process(target=run_experiment, args=(seed,)).start()
