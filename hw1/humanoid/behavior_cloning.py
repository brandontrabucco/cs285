"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import multiprocessing
from cs285.baselines.imitate.behavior_cloning import BehaviorCloning
from gym.envs.mujoco.humanoid import HumanoidEnv


def run_experiment(experiment_id):
    BehaviorCloning(
        HumanoidEnv,
        logging_dir="./humanoid/behavior_cloning/{}".format(experiment_id),
        hidden_size=256,
        num_hidden_layers=2,
        exploration_noise_std=0.1,
        expert_policy_ckpt="./humanoid/expert_policy.ckpt",
        num_threads=10,
        max_path_length=1000,
        max_num_steps=1000000,
        batch_size=256,
        num_epochs=1000,
        num_episodes_per_epoch=0,
        num_trains_per_epoch=10,
        num_episodes_before_train=10,
        num_epochs_per_eval=1,
        num_episodes_per_eval=10).launch()


if __name__ == "__main__":

    num_seeds = 5
    for seed in range(num_seeds):
        multiprocessing.Process(target=run_experiment, args=(seed,)).start()
