"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import multiprocessing
from cs285.baselines.on_policy.policy_gradient import PolicyGradient
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


def run_experiment(experiment_id):
    PolicyGradient(
        HalfCheetahEnv,
        logging_dir="./half_cheetah/policy_gradient/{}".format(experiment_id),
        hidden_size=256,
        num_hidden_layers=2,
        max_num_paths=333,
        num_threads=10,
        max_path_length=150,
        reward_scale=1.0,
        discount=0.99,
        policy_learning_rate=0.02,
        batch_size=333,
        num_epochs=10000,
        num_episodes_per_epoch=333,
        num_trains_per_epoch=1,
        num_episodes_before_train=0,
        num_epochs_per_eval=1,
        num_episodes_per_eval=20).launch()


if __name__ == "__main__":

    num_seeds = 5
    for seed in range(num_seeds):
        multiprocessing.Process(target=run_experiment, args=(seed,)).start()
