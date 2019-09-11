"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285 import nested_apply
import numpy as np
import os
import pickle as pkl


class ReplayBuffer(object):

    def __init__(
            self,
            max_path_length=1000,
            max_num_paths=1000,
            selector=None,
            monitor=None,
            logging_prefix=""
    ):
        self.max_path_length = max_path_length
        self.max_num_paths = max_num_paths
        self.selector = selector if selector is not None else (lambda x: x)
        self.monitor = monitor
        self.logging_prefix = logging_prefix
        self.head = 0
        self.size = 0
        self.total_paths = 0
        self.total_steps = 0
        self.tail = np.zeros([self.max_num_paths], dtype=np.int32)
        self.observations = None
        self.actions = None
        self.rewards = None

    def inflate_backend(
            self,
            x
    ):
        # used to initialize the replay buffer storage
        return np.zeros([
            self.max_num_paths,
            self.max_path_length,
            *x.shape], dtype=np.float32)

    def inflate(
            self,
            observation,
            action,
            reward
    ):
        # initialize the replay buffer storage of paths
        self.observations = nested_apply(self.inflate_backend, observation)
        self.actions = self.inflate_backend(action)
        self.rewards = self.inflate_backend(reward)

        if self.monitor is not None:
            self.monitor.record(self.logging_prefix + "head", 0)
            self.monitor.record(self.logging_prefix + "size", 0)
            self.monitor.record(self.logging_prefix + "total_paths", 0)
            self.monitor.record(self.logging_prefix + "total_steps", 0)

    def insert_path_backend(
            self,
            structure,
            path
    ):
        # insert samples along a path into the replay buffer
        for tail, y in enumerate(path[:self.max_path_length]):
            structure[self.head, tail, ...] = y

    def insert_path(
            self,
            observations,
            actions,
            rewards
    ):
        # check if the replay buffer has not been initialized
        if self.observations is None or self.actions is None or self.rewards is None:
            self.inflate(observations[0], actions[0], rewards[0])

        # insert samples along a path into the replay buffer
        for tail, (x, y, z) in enumerate(zip(
                observations[:self.max_path_length],
                actions[:self.max_path_length],
                rewards[:self.max_path_length])):

            # necessary for the nested observation dictionary
            def insert_backend(structure, data):
                structure[self.head, tail, ...] = data

            # assign all steps into the buffer
            nested_apply(insert_backend, self.observations, x)
            insert_backend(self.actions, y)
            insert_backend(self.rewards, z)

        # keep track of the length of a path
        self.tail[self.head] = len(rewards[:self.max_path_length])

        # keep track of the size of the replay buffer and the next writable slot
        self.head = (self.head + 1) % self.max_num_paths
        self.size = min(self.size + 1, self.max_num_paths)
        self.total_paths += 1
        self.total_steps += len(rewards[:self.max_path_length])

        if self.monitor is not None:
            self.monitor.record(self.logging_prefix + "head", self.head)
            self.monitor.record(self.logging_prefix + "size", self.size)
            self.monitor.record(self.logging_prefix + "total_paths", self.total_paths)
            self.monitor.record(self.logging_prefix + "total_steps", self.total_steps)

    def sample_paths(
            self,
            batch_size
    ):
        # sample entire paths from the replay buffer (for on policy methods)
        idx = np.random.choice(
            self.size,
            size=batch_size,
            replace=(self.size < batch_size))

        # note that observations can be dicts, tuples, sets, or lists
        observations = nested_apply(
            lambda x: x[idx, ...],
            self.selector(self.observations))

        # actions and rewards must be vectors and scalars respectively
        actions = self.actions[idx, ...]
        rewards = self.rewards[idx, :]

        # terminals indicates which states are valid states (1 if valid)
        terminals = (np.arange(self.max_path_length)[np.newaxis, :] <
                     self.tail[idx][:, np.newaxis]).astype(np.float32)
        return observations, actions, rewards * terminals, terminals

    def sample_steps(
            self,
            batch_size
    ):
        # sample transitions from the replay buffer (for off policy methods)
        positions = np.arange(self.max_path_length)

        # keep track of which steps are valid and can be sampled using a mask
        ps = (positions[:, np.newaxis] <
              self.tail[np.newaxis, :]).astype(np.float32)

        # enumerate all steps available in the replay buffer
        candidate_steps = np.stack(
            np.meshgrid(np.arange(self.max_num_paths), positions),
            axis=(-1)).reshape(-1, 2)

        # take indices into the replay buffer of the samples transitions
        idx = np.take(candidate_steps, np.random.choice(
            self.max_num_paths * self.max_path_length,
            size=batch_size,
            replace=(self.size * self.max_path_length < batch_size),
            p=(ps.reshape(-1) / ps.sum())), axis=0)

        # note that observations can be dicts, tuples, sets, or lists
        observations = nested_apply(
            lambda x: x[idx[:, 0], idx[:, 1], ...],
            self.selector(self.observations))

        # actions and rewards must be vectors and scalars respectively
        actions = self.actions[idx[:, 0], idx[:, 1], ...]
        rewards = self.rewards[idx[:, 0], idx[:, 1]]
        clipped_next_idx = np.minimum(idx[:, 1] + 1, self.max_path_length - 1)
        next_observations = nested_apply(
            lambda x: x[idx[:, 0], clipped_next_idx, ...],
            self.selector(self.observations))

        # terminals indicate which states are final (1 if not final)
        terminals = (positions[np.newaxis, :] + 1 <
                     self.tail[:, np.newaxis]).astype(
            np.float32)[idx[:, 0], idx[:, 1]]
        return observations, actions, rewards, next_observations, terminals

    def save(
        self,
        logging_dir
    ):
        # save the replay buffer to disk
        with open(os.path.join(
                logging_dir,
                self.logging_prefix + "replay.buffer"), "wb") as f:
            # convert replay buffer into a pickle able form
            state = dict(
                total_paths=self.total_paths,
                total_steps=self.total_steps,
                size=self.size,
                head=self.head,
                tail=self.tail,
                observations=self.observations,
                actions=self.actions,
                rewards=self.rewards)

            # save the replay buffer to a file
            pkl.dump(state, f)

    def load(
        self,
        logging_dir
    ):
        # load the replay buffer from disk if it exists
        replay_path = os.path.join(
            logging_dir, self.logging_prefix + "replay.buffer")
        if os.path.exists(replay_path):
            with open(replay_path,  "rb") as f:
                state = pkl.load(f)

                # assign state of loaded buffer to the current buffer
                self.total_paths = state["total_paths"]
                self.total_steps = state["total_steps"]
                self.size = state["size"]
                self.head = state["head"]
                self.tail = state["tail"]
                self.observations = state["observations"]
                self.actions = state["actions"]
                self.rewards = state["rewards"]
