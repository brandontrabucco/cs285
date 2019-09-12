"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod
import os
import pickle as pkl


class ReplayBuffer(ABC):

    def __init__(
            self,
            monitor=None,
            logging_prefix=""
    ):
        self.monitor = monitor
        self.logging_prefix = logging_prefix

        # storage structures for the samples collected
        self.observations = None
        self.actions = None
        self.rewards = None
        self.terminals = None

        # parameters to indicate the size of the buffer
        self.head = 0
        self.size = 0
        self.total_steps = 0
        self.total_paths = 0

    def log(
            self
    ):
        if self.monitor is not None:
            # record the current size of the buffer
            self.monitor.record(self.logging_prefix + "head", self.head)
            self.monitor.record(self.logging_prefix + "size", self.size)

            # record the total amount of samples collected
            self.monitor.record(self.logging_prefix + "total_steps", self.total_steps)
            self.monitor.record(self.logging_prefix + "total_paths", self.total_paths)

    def save(
        self,
        logging_dir
    ):
        # save the replay buffer to disk
        replay_path = os.path.join(logging_dir, self.logging_prefix + "replay.buffer")
        os.makedirs(os.path.dirname(replay_path))
        with open(replay_path, "wb") as f:
            state = dict(
                observations=self.observations,
                actions=self.actions,
                rewards=self.rewards,
                terminals=self.terminals,
                size=self.size,
                head=self.head,
                total_steps=self.total_steps,
                total_paths=self.total_paths)
            pkl.dump(state, f)

    def load(
        self,
        logging_dir
    ):
        # load the replay buffer from disk if it exists
        replay_path = os.path.join(logging_dir, self.logging_prefix + "replay.buffer")
        if os.path.exists(replay_path):
            with open(replay_path,  "rb") as f:
                state = pkl.load(f)
            self.observations = state["observations"]
            self.actions = state["actions"]
            self.rewards = state["rewards"]
            self.terminals = state["terminals"]
            self.size = state["size"]
            self.head = state["head"]
            self.total_steps = state["total_steps"]
            self.total_paths = state["total_paths"]

    @abstractmethod
    def insert_path(
            self,
            observations,
            actions,
            rewards
    ):
        return NotImplemented

    @abstractmethod
    def sample(
            self,
            batch_size
    ):
        return NotImplemented
