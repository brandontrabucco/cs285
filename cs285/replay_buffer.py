"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285 import nested_apply
import numpy as np


class ReplayBuffer(object):

    def __init__(
            self,
            max_path_length=1000,
            max_num_paths=1000,
            selector=None
    ):
        self.max_path_length = max_path_length
        self.max_num_paths = max_num_paths
        self.selector = selector if selector is None else (lambda x: x)
        self.size = 0
        self.head = 0
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

    def insert_path_backend(
            self,
            structure,
            path
    ):
        # insert samples along a path into the replay buffer
        for tail, y in enumerate(path[:self.max_path_length]):
            structure[self.head, tail, ...] = y

        # keep track of the length of a path
        self.tail[self.head] = len(path[:self.max_path_length])

        # keep track of the size of the replay buffer and the next writable slot
        self.head = (self.head + 1) % self.max_num_paths
        self.size = min(self.size + 1, self.max_num_paths)

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
        nested_apply(self.insert_path_backend, self.observations, observations)

        # actions and rewards must be vectors and scalars respectively
        self.insert_path_backend(self.actions, actions)
        self.insert_path_backend(self.rewards, rewards)

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
        ps = (positions[:, np.newaxis] < self.tail[np.newaxis, :]).astype(np.float32)

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
        next_observations = nested_apply(
            lambda x: x[idx[:, 0], np.minimum(idx[:, 1] + 1, self.max_path_length), ...],
            self.selector(self.observations))

        # these terminals indicate which states are terminal states (1 if non terminal)
        terminals = (positions[np.newaxis, :] + 1 <
                     self.tail[:, np.newaxis]).astype(np.float32)[idx[:, 0], idx[:, 1]]
        return observations, actions, rewards, next_observations, terminals
