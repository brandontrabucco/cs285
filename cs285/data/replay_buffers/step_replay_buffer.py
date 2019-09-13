"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285 import nested_apply
from cs285.data.replay_buffers.replay_buffer import ReplayBuffer
import numpy as np


class StepReplayBuffer(ReplayBuffer):

    def __init__(
            self,
            max_num_steps=1000,
            selector=None,
            **kwargs
    ):
        ReplayBuffer.__init__(self, **kwargs)

        # parameters to control how the buffer is created and managed
        self.max_num_steps = max_num_steps
        self.selector = selector if selector is not None else (lambda x: x)

    def inflate_backend(
            self,
            x
    ):
        # create numpy arrays to store samples
        return np.zeros([self.max_num_steps, *x.shape], dtype=np.float32)

    def insert_backend(
            self,
            structure,
            data
    ):
        # insert samples into the numpy array
        structure[self.head, ...] = data

    def insert_path(
            self,
            observations,
            actions,
            rewards
    ):
        # insert a path into the replay buffer
        self.total_paths += 1
        observations = observations[:self.max_num_steps]
        actions = actions[:self.max_num_steps]
        rewards = rewards[:self.max_num_steps]

        # inflate the replay buffer if not inflated
        if self.observations is None:
            self.observations = nested_apply(self.inflate_backend, observations[0])
            self.actions = self.inflate_backend(actions[0])
            self.rewards = self.inflate_backend(rewards[0])
            self.terminals = self.inflate_backend(rewards[0])

        # insert all samples into the buffer
        for i, (o, a, r) in enumerate(zip(observations, actions, rewards)):
            nested_apply(self.insert_backend, self.observations, o)
            self.insert_backend(self.actions, a)
            self.insert_backend(self.rewards, r)
            self.insert_backend(self.terminals, 1.0 if i < len(observations) - 1 else 0.0)

            # increment the head and size
            self.head = (self.head + 1) % self.max_num_steps
            self.size = min(self.size + 1, self.max_num_steps)
            self.total_steps += 1

    def sample(
            self,
            batch_size
    ):
        # determine which steps to sample from
        idx = np.random.choice(self.size, size=batch_size, replace=(self.size < batch_size))
        next_idx = (idx + 1) % self.max_num_steps

        def sample(data):
            return data[idx, ...]

        def sample_next(data):
            return data[next_idx, ...]

        # sample current batch from a nested samplers structure
        observations = nested_apply(sample, self.selector(self.observations))
        actions = sample(self.actions)
        rewards = sample(self.rewards)
        next_observations = nested_apply(sample_next, self.selector(self.observations))
        terminals = sample(self.terminals)

        # return the samples in a batch
        return observations, actions, rewards, next_observations, terminals
