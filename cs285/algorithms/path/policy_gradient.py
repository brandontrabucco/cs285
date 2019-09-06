"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.algorithms.path.path_algorithm import PathAlgorithm
from cs285 import discounted_sum
import tensorflow as tf


class SAC(PathAlgorithm):

    def __init__(
        self,
        policy,
        reward_scale=1.0,
        discount=0.99,
        policy_optimizer_class=tf.keras.optimizers.Adam,
        policy_optimizer_kwargs=None,
        **kwargs,
    ):
        PathAlgorithm.__init__(self, **kwargs)
        self.policy = policy

        self.reward_scale = reward_scale
        self.discount = discount

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
        for i in range(self.off_policy_updates):
            with tf.GradientTape() as tape:

                # COMPUTE ADVANTAGES
                discounted_returns = discounted_sum(rewards, self.discount)
                self.record("discounted_returns", tf.reduce_mean(discounted_returns))
                advantages = discounted_returns - tf.reduce_mean(discounted_returns)
                self.record("advantages", tf.reduce_mean(advantages))

                # COMPUTE SURROGATE POLICY LOSS
                policy_log_prob = self.policy.log_prob(actions, observations) * advantages
                self.record("policy_ratio", tf.reduce_mean(policy_log_prob))
                policy_loss = -tf.reduce_mean(policy_log_prob)
                self.record("policy_loss", tf.reduce_mean(policy_loss))

            # BACK PROP GRADIENTS
            policy_gradients = tape.gradient(policy_loss, self.policy.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(
                policy_gradients, self.policy.trainable_variables))
