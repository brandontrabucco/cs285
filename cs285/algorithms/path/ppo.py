"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.algorithms.path.path_algorithm import PathAlgorithm
from cs285 import discounted_sum
import tensorflow as tf


class SAC(PathAlgorithm):

    def __init__(
        self,
        policy,
        old_policy,
        vf,
        reward_scale=1.0,
        discount=0.99,
        epsilon=0.2,
        lamb=0.95,
        off_policy_updates=1,
        critic_updates=1,
        vf_optimizer_class=tf.keras.optimizers.Adam,
        vf_optimizer_kwargs=None,
        policy_optimizer_class=tf.keras.optimizers.Adam,
        policy_optimizer_kwargs=None,
        **kwargs,
    ):
        PathAlgorithm.__init__(self, **kwargs)
        self.policy = policy
        self.old_policy = old_policy
        self.vf = vf

        self.reward_scale = reward_scale
        self.discount = discount
        self.epsilon = epsilon
        self.lamb = lamb
        self.off_policy_updates = off_policy_updates
        self.critic_updates = critic_updates

        if vf_optimizer_kwargs is None:
            vf_optimizer_kwargs = dict(lr=0.001, clipnorm=1.0)
        self.vf_optimizer = vf_optimizer_class(**vf_optimizer_kwargs)

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
        for i in range(self.critic_updates):
            with tf.GradientTape() as tape:

                # COMPUTE VALUE FUNCTION LOSS
                discounted_returns = discounted_sum(rewards, self.discount)
                self.record("discounted_returns", tf.reduce_mean(discounted_returns))
                values = self.vf(observations)[..., 0]
                self.record("values", tf.reduce_mean(values))
                vf_loss = tf.losses.mean_squared_error(discounted_returns, values)
                self.record("vf_loss", tf.reduce_mean(vf_loss))

            # BACK PROP GRADIENTS
            vf_gradients = tape.gradient(vf_loss, self.vf.trainable_variables)
            self.qf_optimizer.apply_gradients(zip(
                vf_gradients, self.vf.trainable_variables))

        self.old_policy.set_weights(self.policy.get_weights())
        for i in range(self.off_policy_updates):
            with tf.GradientTape() as tape:

                # COMPUTE GENERALIZED ADVANTAGES
                delta_v = (rewards - values +
                           self.discount * tf.pad(values, [[0, 0], [0, 1]])[:, 1:])
                self.record("delta_v", tf.reduce_mean(delta_v))
                advantages = discounted_sum(delta_v, self.discount * self.lamb)
                self.record("advantages", tf.reduce_mean(advantages))

                # COMPUTE SURROGATE POLICY LOSS
                policy_ratio = tf.exp(self.policy.log_prob(actions, observations) -
                                      self.old_policy.log_prob(actions, observations))
                self.record("policy_ratio", tf.reduce_mean(policy_ratio))
                policy_loss = -tf.reduce_mean(
                    tf.minimum(
                        policy_ratio * advantages,
                        tf.clip_by_value(
                            policy_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages))
                self.record("policy_loss", tf.reduce_mean(policy_loss))

            # BACK PROP GRADIENTS
            policy_gradients = tape.gradient(policy_loss, self.policy.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(
                policy_gradients, self.policy.trainable_variables))
