"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf
import os


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
                include_optimizer=False)

        # save the replay buffer to the disk
        if self.replay_buffer is not None:
            self.replay_buffer.save(self.logging_dir)

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
            self.replay_buffer.load(self.logging_dir)
