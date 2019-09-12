"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf
import numpy as np
import argparse
import pickle as pkl
import os


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
    parser.add_argument('--observation_low', nargs='+', type=float, default=[])
    parser.add_argument('--observation_high', nargs='+', type=float, default=[])
    parser.add_argument('--action_low', nargs='+', type=float, default=[])
    parser.add_argument('--action_high', nargs='+', type=float, default=[])
    args = parser.parse_args()

    # create the directory to house the replay buffer
    tf.io.gfile.makedirs(os.path.dirname(args.output_policy_file))

    # load the expert policy from a file
    with tf.io.gfile.GFile(args.expert_policy_file, "rb") as f:
        data = pkl.load(f)

    # process the architecture params
    activation_type_type = data['nonlin_type']
    policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]
    policy_params = data[policy_type]

    # load the samplers normalizer for this policy
    norm_mean = policy_params['obsnorm']['Standardizer']['mean_1_D']
    norm_mean_sq = policy_params['obsnorm']['Standardizer']['meansq_1_D']
    norm_std = np.sqrt(np.maximum(0, norm_mean_sq - np.square(norm_mean)))

    # create the observation lower bound
    if len(args.observation_low) == 0:
        observation_low = -np.ones_like(norm_mean)
    elif len(args.observation_low) == 1:
        observation_low = np.full_like(norm_mean, args.observation_low[0])
    else:
        observation_low = np.array(args.observation_low)

    # create the observation upper bound
    if len(args.observation_high) == 0:
        observation_high = np.ones_like(norm_mean)
    elif len(args.observation_low) == 1:
        observation_high = np.full_like(norm_mean, args.observation_high[0])
    else:
        observation_high = np.array(args.observation_high)

    # define the input shape
    obs_bo = tf.keras.layers.Input(shape=norm_mean.shape[-1])

    # de normalize the observations entering the network
    current_bo = tf.keras.layers.Lambda(
        lambda x: observation_low + (x + 1.0) * 0.5 * (observation_high - observation_low))(obs_bo)

    # build the policy architecture in keras
    current_bo = tf.keras.layers.Lambda(
        lambda x: (x - norm_mean) / (norm_std + 1e-6))(current_bo)
    layer_params = policy_params['hidden']['FeedforwardNet']
    for layer_name in sorted(layer_params.keys()):
        current_bo = create_keras_layer(
            current_bo, layer_params[layer_name], activation_type_type)
    output_bo = create_keras_layer(
        current_bo, policy_params['out'], "linear")

    # create the action lower bound
    if len(args.action_low) == 0:
        action_low = -np.ones(output_bo.shape[1:])
    elif len(args.action_low) == 1:
        action_low = np.full(output_bo.shape[1:], args.action_low[0])
    else:
        action_low = np.array(args.action_low)

    # create the action upper bound
    if len(args.action_high) == 0:
        action_high = np.ones(output_bo.shape[1:])
    elif len(args.action_high) == 1:
        action_high = np.full(output_bo.shape[1:], args.action_high[0])
    else:
        action_high = np.array(args.action_high)

    # normalize the actions exiting the network
    output_bo = tf.keras.layers.Lambda(
        lambda x: (x - action_low) * 2.0 / (action_high - action_low) - 1.0)(output_bo)

    # create a keras model
    model = tf.keras.models.Model(inputs=obs_bo, outputs=output_bo)

    # save the keras model to the disk
    tf.keras.models.save_model(
        model,
        args.output_policy_file,
        overwrite=True,
        include_optimizer=False)
