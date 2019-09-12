"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


class Relabeler(object):

    def __init__(
            self,
            replay_buffer
    ):
        # relabel samples with some probability
        self.replay_buffer = replay_buffer

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

    def relabel_sample(
            self,
            *args
    ):
        # relabel paths as they exit the replay buffer
        return args

    def sample(
            self,
            batch_size
    ):
        # relabel paths as they exit the replay buffer
        return self.relabel_sample(
            *self.replay_buffer.sample(batch_size))
