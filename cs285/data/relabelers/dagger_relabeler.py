"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.data.relabelers import Relabeler
import tensorflow as tf


class DaggerRelabeler(Relabeler):

    def __init__(
            self,
            expert_policy,
            *args,
            expert_selector=None,
            **kwargs
    ):
        # relabel samples with expert actions with some probability
        Relabeler.__init__(self, *args, **kwargs)
        self.expert_policy = expert_policy
        self.expert_selector = (
            expert_selector if expert_selector is not None else (lambda x: x))

    def relabel_insert_path(
            self,
            observations,
            actions,
            rewards
    ):
        # compute the expert actions across the episode
        expert_actions, log_pi = self.expert_policy.expected_value(
            tf.stack([self.expert_selector(x)
                      for x in observations], axis=0))

        # then return the samples in an original format to enter the buffer
        return observations, tf.unstack(expert_actions, axis=0), rewards
