"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285 import nested_apply
from cs285.core.replay_buffers.replay_buffer import ReplayBuffer
import numpy as np


class PathReplayBuffer(ReplayBuffer):

    def __init__(
            self,
            max_path_length=1000,
            max_num_paths=1000,
            selector=None,
            **kwargs
    ):
        ReplayBuffer.__init__(self, **kwargs)

        self.max_path_length = max_path_length
        self.max_num_paths = max_num_paths
        self.selector = selector if selector is not None else (lambda x: x)

    def inflate_backend(
            self,
            x
    ):
        # create numpy arrays to store samples
        return np.zeros([self.max_num_paths, self.max_path_length, *x.shape], dtype=np.float32)

    def insert_backend(
            self,
            structure,
            data
    ):
        # insert samples into the numpy array
        structure[self.head, int(self.terminals[self.head]), ...] = data

    def insert_path(
            self,
            observations,
            actions,
            rewards
    ):
        # insert a path into the replay buffer
        self.total_paths += 1
        observations = observations[:self.max_path_length]
        actions = actions[:self.max_path_length]
        rewards = rewards[:self.max_path_length]

        # inflate the replay buffer if not inflated
        if self.observations is None:
            self.observations = nested_apply(self.inflate_backend, observations[0])
            self.actions = self.inflate_backend(actions[0])
            self.rewards = self.inflate_backend(rewards[0])
            self.terminals = self.inflate_backend(rewards[0])

        # insert all samples into the buffer
        for i, (o, a, r) in zip(observations, actions, rewards):
            nested_apply(self.insert_backend, self.observations, o)
            self.insert_backend(self.actions, a)
            self.insert_backend(self.rewards, r)
            self.insert_backend(self.terminals, i + 1)
            self.total_steps += 1

        # increment the head and size
        self.head = (self.head + 1) % self.max_path_length
        self.size = min(self.size + 1, self.max_path_length)

    def sample(
            self,
            batch_size
    ):
        # determine which steps to sample from
        idx = np.random.choice(self.size, size=batch_size, replace=(self.size < batch_size))

        def sample(data):
            return data[idx, ...]

        # sample current batch from a nested samplers structure
        observations = nested_apply(sample, self.selector(self.observations))
        actions = sample(self.actions)
        rewards = sample(self.rewards)
        terminals = (np.arange(self.max_path_length) < sample(self.terminals)).astype(np.float32)

        # return the samples in a batch
        return observations, actions, rewards, terminals
