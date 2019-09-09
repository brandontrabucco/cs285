"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf
import os
import pickle as pkl


class Saver(object):

    def __init__(
        self,
        logging_dir,
        replay_buffer=None,
        **models
    ):
        # create the folder in which the saved models are placed
        tf.io.gfile.makedirs(logging_dir)
        self.logging_dir = logging_dir
        self.replay_buffer = replay_buffer
        self.models = models

    def save(
        self
    ):
        # save the models using the keras checkpoint system
        for name, model in self.models.items():
            tf.keras.models.save_model(
                model,
                os.path.join(self.logging_dir, name + ".ckpt"),
                overwrite=True,
                include_optimizer=True)

        # save the replay buffer to the disk
        if self.replay_buffer is not None:
            with open(os.path.join(self.logging_dir, "replay.buffer"),  "wb") as f:

                # convert replay buffer into a pickle able form
                replay_buffer = dict(
                    size=self.replay_buffer.size,
                    head=self.replay_buffer.head,
                    tail=self.replay_buffer.tail,
                    observations=self.replay_buffer.observations,
                    actions=self.replay_buffer.actions,
                    rewards=self.replay_buffer.rewards)

                # save the replay buffer to a file
                pkl.dump(replay_buffer, f)

    def load(
        self
    ):
        # load the models using the keras checkpoint system
        for name, model in self.models.items():
            model_path = os.path.join(self.logging_dir, name + ".ckpt")
            if tf.io.gfile.exists(model_path):
                existing_model = tf.keras.models.load_model(model_path, compile=False)
                model.set_weights(existing_model.get_weights())

        # load the replay buffer from the disk
        if self.replay_buffer is not None:
            replay_path = os.path.join(self.logging_dir, "replay.buffer")
            if tf.io.gfile.exists(replay_path):
                with open(replay_path,  "rb") as f:
                    replay_buffer = pkl.load(f)

                    # assign the state of the loaded replay buffer to the current replay buffer
                    self.replay_buffer.size = replay_buffer["size"]
                    self.replay_buffer.head = replay_buffer["head"]
                    self.replay_buffer.tail = replay_buffer["tail"]
                    self.replay_buffer.observations = replay_buffer["observations"]
                    self.replay_buffer.actions = replay_buffer["actions"]
                    self.replay_buffer.rewards = replay_buffer["rewards"]
