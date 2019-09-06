"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from tensorboard import program
import os
import queue
import threading
import tensorflow as tf
import io
import matplotlib.pyplot as plt


def plot_to_tensor(
    xs,
    ys,
    title,
    x_label,
    y_label
):
    plt.clf()
    for xxss, yyss in zip(xs, ys):
        plt.plot(xxss, yyss)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return tf.expand_dims(tf.image.decode_png(buffer.getvalue(), channels=4), 0)


def create_and_listen(
    logging_dir,
    record_queue
):
    tf.io.gfile.makedirs(logging_dir)
    writer = tf.summary.create_file_writer(logging_dir)
    tf.summary.experimental.set_step(0)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logging_dir])

    try:
        url = tb.launch()
        print("TensorBoard launched at: {}".format(url))
    except program.TensorBoardServerException:
        print("TensorBoard failed to launch")

    while True:
        if not record_queue.empty():
            step, key, value = record_queue.get()
            tf.summary.experimental.set_step(step)

            with writer.as_default():
                if len(tf.shape(value)) == 0:
                    tf.summary.scalar(key, value)

                elif len(tf.shape(value)) == 1:
                    splits = key.split(",")
                    tf.summary.image(splits[0], plot_to_tensor(
                        tf.expand_dims(tf.range(tf.shape(value)[1]), 0),
                        tf.expand_dims(value, 0), splits[0], splits[1], splits[2]))

                elif len(tf.shape(value)) == 2:
                    splits = key.split(",")
                    tf.summary.image(splits[0], plot_to_tensor(
                        tf.tile(tf.expand_dims(
                            tf.range(tf.shape(value)[1]), 0), [tf.shape(value)[0], 1]),
                        value, splits[0], splits[1], splits[2]))

                elif len(tf.shape(value)) == 3:
                    tf.summary.image(key, tf.expand_dims(value, 0) * 0.5 + 0.5)

                elif len(tf.shape(value)) == 4:
                    tf.summary.image(key, value * 0.5 + 0.5)

                else:
                    tf.summary.scalar(key, value)


class Monitor(object):

    def __init__(
        self,
        logging_dir
    ):
        self.logging_dir = logging_dir
        self.record_queue = queue.Queue()
        self.thread = threading.Thread(
            target=create_and_listen, args=(logging_dir, self.record_queue))
        self.thread.start()
        self.step_file = os.path.join(logging_dir, "step")
        if tf.io.gfile.exists(self.step_file):
            with tf.io.gfile.GFile(self.step_file, "r") as f:
                self.step = int(f.read().strip())
        else:
            self.step = 0
        self.lock = threading.Lock()

    def increment(
        self
    ):
        self.lock.acquire()
        self.step += 1
        self.lock.release()

    def save_step(
        self
    ):
        self.lock.acquire()
        with tf.io.gfile.GFile(self.step_file, "w") as f:
            f.write(str(self.step))
        self.lock.release()

    def record(
        self,
        key,
        value,
    ):
        self.lock.acquire()
        self.record_queue.put((self.step, key, value))
        self.lock.release()
