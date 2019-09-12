"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from tensorboard import program
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


def create_monitor_thread(
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

    # then loop forever and save samplers via the queue to tensor board
    while True:

        # pull the next samplers elements to log from the queue
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
