"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


class Relabeler(object):

    def __init__(
            self,
            replay_buffer,
            relabel_probability=1.0
    ):
        # relabel samples with some probability
        self.replay_buffer = replay_buffer
        self.relabel_probability = relabel_probability

    def __setattr__(
        self,
        attr,
        value
    ):
        # pass attribute assignments to the replay buffer
        if not attr == "replay_buffer":
            setattr(self.replay_buffer, attr, value)
        else:
            self.__dict__[attr] = value

    def __getattr__(
        self,
        attr
    ):
        # pass attribute lookups to the replay buffer
        if not attr == "replay_buffer":
            return getattr(self.replay_buffer, attr)
        else:
            return self.__dict__[attr]

    def relabel_insert_path(
            self,
            observations,
            actions,
            rewards
    ):
        # relabel samples as they enter the replay buffer
        return observations, actions, rewards

    def insert_path(
            self,
            observations,
            actions,
            rewards
    ):
        # relabel samples as they enter the replay buffer
        return self.replay_buffer.insert_path(
            *self.relabel_insert_path(observations, actions, rewards))

    def relabel_sample_paths(
            self,
            observations,
            actions,
            rewards,
            terminals
    ):
        # relabel paths as they exit the replay buffer
        return observations, actions, rewards, terminals

    def sample_paths(
            self,
            batch_size
    ):
        # relabel paths as they exit the replay buffer
        return self.relabel_sample_paths(
            *self.replay_buffer.sample_paths(batch_size))

    def relabel_sample_steps(
            self,
            observations,
            actions,
            rewards,
            next_observations,
            terminals
    ):
        # relabel steps as they exit the replay buffer
        return observations, actions, rewards, next_observations, terminals

    def sample_steps(
            self,
            batch_size
    ):
        # relabel steps as they exit the replay buffer
        return self.relabel_sample_steps(
            *self.replay_buffer.sample_steps(batch_size))

    def relabel_mask(
            self,
            data
    ):
        # generate a bernoulli mask that determines whether an element is relabeled
        relabel_condition = tf.math.less_equal(
            tf.random.uniform(
                tf.shape(data)[:2],
                maxval=1.0,
                dtype=tf.float32), self.relabel_probability)

        # expand to the same shape as the data
        while len(relabel_condition.shape) < len(data.shape):
            relabel_condition = tf.expand_dims(relabel_condition, -1)

        # broadcast to data shape
        return tf.broadcast_to(relabel_condition, tf.shape(data))
