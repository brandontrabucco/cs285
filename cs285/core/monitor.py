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
    # open a new matplotlib plot figure
    plt.clf()

    # plot several lines on the newly spawned figure
    for xxss, yyss in zip(xs, ys):
        plt.plot(xxss, yyss)

    # configure the plot with names
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # convert the plot to jpg bytes for tensor board
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # convert the bytes to an image tensor
    return tf.expand_dims(tf.image.decode_png(buffer.getvalue(), channels=4), 0)


def create_and_listen(
    logging_dir,
    record_queue
):
    # create the tensor board server which can be logged into on localhost
    tf.io.gfile.makedirs(logging_dir)
    writer = tf.summary.create_file_writer(logging_dir)
    tf.summary.experimental.set_step(0)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logging_dir])

    # attempt to launch the tensor board server
    try:
        url = tb.launch()
        print("TensorBoard launched at: {}".format(url))
    except program.TensorBoardServerException:
        print("TensorBoard failed to launch")

    # then loop forever and save data via the queue to tensor board
    while True:

        # pull the next data elements to log from the queue
        if not record_queue.empty():
            step, key, value = record_queue.get()
            tf.summary.experimental.set_step(step)

            with writer.as_default():

                # generate a plot and write the plot to tensor board
                if len(tf.shape(value)) == 1:
                    splits = key.split(",")
                    tf.summary.image(splits[0], plot_to_tensor(
                        tf.expand_dims(tf.range(tf.shape(value)[1]), 0),
                        tf.expand_dims(value, 0), splits[0], splits[1], splits[2]))

                # generate several plots and write the plot to tensor board
                elif len(tf.shape(value)) == 2:
                    splits = key.split(",")
                    tf.summary.image(splits[0], plot_to_tensor(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(value)[1]), 0), [tf.shape(value)[0], 1]),
                        value, splits[0], splits[1], splits[2]))

                # write a single image to tensor board
                elif len(tf.shape(value)) == 3:
                    tf.summary.image(key, tf.expand_dims(value, 0) * 0.5 + 0.5)

                # write several images to tensor board
                elif len(tf.shape(value)) == 4:
                    tf.summary.image(key, value * 0.5 + 0.5)

                # otherwise, assume the tensor is still a scalar
                else:
                    tf.summary.scalar(key, value)


class Monitor(object):

    def __init__(
        self,
        logging_dir
    ):
        # create a separate tensor board logging thread
        self.logging_dir = logging_dir

        # communicate with the thread via a queue
        self.record_queue = queue.Queue()

        # listen on the specified thread in a loop
        self.thread = threading.Thread(
            target=create_and_listen, args=(logging_dir, self.record_queue))
        self.thread.start()
        
        # keep track of the current global step, and resume training if necessary
        self.step_file = os.path.join(logging_dir, "step")
        if tf.io.gfile.exists(self.step_file):
            with tf.io.gfile.GFile(self.step_file, "r") as f:
                self.step = int(f.read().strip())

        # otherwise start from scratch
        else:
            self.step = 0
        self.lock = threading.Lock()

    def increment(
        self
    ):
        # increment how many steps have been collected so far
        self.lock.acquire()
        self.step += 1
        self.lock.release()

    def save_step(
        self
    ):
        # save the current step (like the tf global step) to a file
        self.lock.acquire()
        with tf.io.gfile.GFile(self.step_file, "w") as f:
            f.write(str(self.step))
        self.lock.release()

    def record(
        self,
        key,
        value,
    ):
        # record a tensor into the tensor board logging thread
        self.lock.acquire()
        self.record_queue.put((self.step, key, value))
        self.lock.release()
