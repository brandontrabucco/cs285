"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.algorithms.path_algorithm import PathAlgorithm
from cs285 import discounted_sum
import tensorflow as tf


class PolicyGradient(PathAlgorithm):

    def __init__(
        self,
        policy,
        reward_scale=1.0,
        discount=0.99,
        policy_optimizer_class=tf.keras.optimizers.Adam,
        policy_optimizer_kwargs=None,
        **kwargs,
    ):
        # train a policy using the vanilla policy gradient
        PathAlgorithm.__init__(self, **kwargs)
        self.policy = policy

        # control the scale and decay of the reward
        self.reward_scale = reward_scale
        self.discount = discount

        # build the optimizer for the policy weights
        if policy_optimizer_kwargs is None:
            policy_optimizer_kwargs = dict(lr=0.0001, clipnorm=1.0)
        self.policy_optimizer = policy_optimizer_class(
            **policy_optimizer_kwargs)

    def update_algorithm(
        self,
        observations,
        actions,
        rewards,
        terminals
    ):
        # update the policy gradient algorithm
        with tf.GradientTape() as tape:

            # COMPUTE ADVANTAGES
            discounted_returns = discounted_sum(rewards, self.discount)
            self.record("discounted_returns", tf.reduce_mean(discounted_returns))
            advantages = discounted_returns - tf.reduce_mean(discounted_returns)
            self.record("advantages", tf.reduce_mean(advantages))

            # COMPUTE SURROGATE POLICY LOSS
            policy_log_prob = self.policy.log_prob(actions, observations)
            self.record("policy_log_prob", tf.reduce_mean(policy_log_prob))
            policy_loss = -tf.reduce_mean(policy_log_prob * advantages)
            self.record("policy_loss", tf.reduce_mean(policy_loss))

        # BACK PROP GRADIENTS
        policy_gradients = tape.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(
            policy_gradients, self.policy.trainable_variables))
