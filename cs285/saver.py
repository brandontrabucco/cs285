"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf
import os


class Saver(object):

    def __init__(
        self,
        logging_dir,
        **models
    ):
        tf.io.gfile.makedirs(logging_dir)
        self.logging_dir = logging_dir
        self.models = models

    def save(
        self
    ):
        for name, model in self.models.items():
            tf.keras.models.save_model(
                model,
                os.path.join(self.logging_dir, name + ".ckpt"),
                overwrite=True,
                include_optimizer=True)

    def load(
        self
    ):
        for name, model in self.models.items():
            model_path = os.path.join(self.logging_dir, name + ".ckpt")
            if tf.io.gfile.exists(model_path):
                existing_model = tf.keras.models.load_model(model_path, compile=False)
                model.set_weights(existing_model.get_weights())
