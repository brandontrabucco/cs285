"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf
import numpy as np
import argparse
import pickle as pkl
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_data_file", type=str)
    parser.add_argument("--replay_buffer_file", type=str)
    parser.add_argument("--max_num_paths", type=int, default=1000)
    parser.add_argument("--max_path_length", type=int, default=1000)
    parser.add_argument('--observation_low', nargs='+', type=float, default=[])
    parser.add_argument('--observation_high', nargs='+', type=float, default=[])
    parser.add_argument('--action_low', nargs='+', type=float, default=[])
    parser.add_argument('--action_high', nargs='+', type=float, default=[])
    args = parser.parse_args()

    # create the directory to house the replay buffer
    tf.io.gfile.makedirs(os.path.dirname(args.replay_buffer_file))

    # load the expert data from a file
    with tf.io.gfile.GFile(args.expert_data_file, "rb") as f:
        x = pkl.load(f)

    # parameters to control the replay buffer
    actual_num_paths = len(x)
    max_num_paths = args.max_num_paths
    max_path_length = args.max_path_length

    # check the initial stats
    observation = x[0]["observation"][0, ...]
    action = x[0]["action"][0, ...]

    # build the raw replay buffer format
    replay_buffer = dict(
        observations=dict(observation=np.zeros([max_num_paths, max_path_length, *observation.shape])),
        actions=np.zeros([max_num_paths, max_path_length, *action.shape]),
        rewards=np.zeros([max_num_paths, max_path_length]),
        size=actual_num_paths,
        head=actual_num_paths,
        tail=np.zeros([max_num_paths]))

    # create the observation lower bound
    if len(args.observation_low) == 0:
        observation_low = -np.ones_like(observation)
    elif len(args.observation_low) == 1:
        observation_low = np.full_like(observation, args.observation_low[0])
    else:
        observation_low = np.array(args.observation_low)

    # create the observation upper bound
    if len(args.observation_high) == 0:
        observation_high = np.ones_like(observation)
    elif len(args.observation_low) == 1:
        observation_high = np.full_like(observation, args.observation_high[0])
    else:
        observation_high = np.array(args.observation_high)

    # create the action lower bound
    if len(args.action_low) == 0:
        action_low = -np.ones_like(action)
    elif len(args.action_low) == 1:
        action_low = np.full_like(action, args.action_low[0])
    else:
        action_low = np.array(args.action_low)

    # create the action upper bound
    if len(args.action_high) == 0:
        action_high = np.ones_like(action)
    elif len(args.action_high) == 1:
        action_high = np.full_like(action, args.action_high[0])
    else:
        action_high = np.array(args.action_high)

    # insert everything into the buffer and normalize the observations and actions
    for i, path in enumerate(x[:max_num_paths]):
        for time_step in range(min(path["observation"].shape[0], max_path_length)):
            replay_buffer["observations"]["observation"][i, time_step, ...] = (
                path["observation"][time_step, ...] - observation_low) * 2.0 / (
                    observation_high - observation_low) - 1.0
            replay_buffer["actions"][i, time_step, ...] = (
                path["action"][time_step, ...] - action_low) * 2.0 / (
                    action_high - action_low) - 1.0
            replay_buffer["rewards"][i, time_step] = path["reward"][time_step, ...]
            replay_buffer["tail"][i] = time_step + 1

    # save the replay buffer to a file
    with tf.io.gfile.GFile(args.replay_buffer_file, "wb") as f:
        pkl.dump(replay_buffer, f)
