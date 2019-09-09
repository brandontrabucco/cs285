"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf
import numpy as np
import argparse
import pickle as pkl


def create_keras_layer(x, layer_type, activation_type):
    W = layer_type['AffineLayer']['W'].astype(np.float32)
    b = layer_type['AffineLayer']['b'].astype(np.float32)

    # create a dense layer with pretrained weights
    x = tf.keras.layers.Dense(
        W.shape[1],
        kernel_initializer=tf.keras.initializers.Constant(W),
        bias_initializer=tf.keras.initializers.Constant(b))(x)

    # create an activation function on top iof the layer
    if activation_type == 'lrelu':
        return tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    else:
        return tf.keras.layers.Activation(activation_type)(x)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--expert_policy_file", type=str)
    parser.add_argument("--output_policy_file", type=str)
    args = parser.parse_args()

    # load the expert policy from a file
    with tf.io.gfile.GFile(args.expert_policy_file, "rb") as f:
        data = pkl.load(f)

    # process the architecture params
    activation_type_type = data['nonlin_type']
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]
    policy_params = data[policy_type]

    # load the data normalizer for this policy
    norm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
    norm_mean_sq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
    norm_std = np.sqrt(np.maximum(0, norm_mean_sq - np.square(norm_mean)))

    # build the policy architecture in keras
    obs_bo = tf.keras.layers.Input(shape=norm_mean.shape)
    current_bo = tf.keras.layers.Lambda(
        lambda x: (x - norm_mean) / (norm_std + 1e-6))(obs_bo)
    layer_params = policy_params['hidden']['FeedforwardNet']
    for layer_name in sorted(layer_params.keys()):
        current_bo = create_keras_layer(
            current_bo, layer_params[layer_name], activation_type_type)
    output_bo = create_keras_layer(
        current_bo, policy_params['out'], "linear")

    # create a keras model
    model = tf.keras.models.Model(inputs=obs_bo, outputs=output_bo)

    # save the keras model to the disk
    tf.keras.models.save_model(
        model,
        args.output_policy_file + ".ckpt",
        overwrite=True,
        include_optimizer=False)
