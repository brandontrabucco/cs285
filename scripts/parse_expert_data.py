"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf
import numpy as np
import argparse
import pickle as pkl


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_data_file", type=str)
    parser.add_argument("--replay_buffer_file", type=str)
    parser.add_argument("--max_num_paths", type=int, default=1000)
    parser.add_argument("--max_path_length", type=int, default=1000)
    args = parser.parse_args()

    tf.io.gfile.makedirs(args.replay_buffer_file)

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

    # insert everything into the buffer
    for i, path in enumerate(x[:max_num_paths]):
        for time_step in range(min(path["observation"].shape[0], max_path_length)):
            replay_buffer["observations"]["observation"][i, time_step, ...] = path["observation"][time_step, ...]
            replay_buffer["actions"][i, time_step, ...] = path["action"][time_step, ...]
            replay_buffer["rewards"][i, time_step] = path["reward"][time_step, ...]
            replay_buffer["tail"][i] = time_step + 1

    # save the replay buffer to a file
    with tf.io.gfile.GFile(args.replay_buffer_file, "wb") as f:
        pkl.dump(replay_buffer, f)
