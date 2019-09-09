"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.relabelers.relabeler import Relabeler
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
        self.expert_selector = expert_selector if expert_selector is not None else (lambda x: x)

    def relabel_insert_path(
            self,
            observations,
            actions,
            rewards
    ):
        # compute the expert actions across the episode
        expert_actions = tf.unstack(
            self.expert_policy(
                tf.stack(self.expert_selector(observations), axis=0)), axis=0)

        # determine which of the samples collected to relabel
        actions = tf.where(self.relabel_mask(actions), expert_actions, actions)

        # then return the samples in an original format to enter the buffer
        return observations, actions, rewards
