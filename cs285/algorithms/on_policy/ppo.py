"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.algorithms.path_algorithm import PathAlgorithm
from cs285 import discounted_sum
import tensorflow as tf


class PPO(PathAlgorithm):

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
        # train a policy using proximal policy optimization
        PathAlgorithm.__init__(self, **kwargs)
        self.policy = policy
        self.old_policy = old_policy
        self.vf = vf

        # control the scale and decay of the reward
        self.reward_scale = reward_scale
        self.discount = discount

        # control how ppo works internally
        self.epsilon = epsilon
        self.lamb = lamb
        self.off_policy_updates = off_policy_updates
        self.critic_updates = critic_updates

        # build an optimizer for the value function weights
        if vf_optimizer_kwargs is None:
            vf_optimizer_kwargs = dict(lr=0.0001, clipnorm=1.0)
        self.vf_optimizer = vf_optimizer_class(
            **vf_optimizer_kwargs)

        # build an optimizer for the policy weights
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
        # train the value function using the discounted return
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

        # train the policy using generalized advantage estimation
        self.old_policy.set_weights(self.policy.get_weights())
        for i in range(self.off_policy_updates):
            with tf.GradientTape() as tape:

                # COMPUTE GENERALIZED ADVANTAGES
                delta_v = (rewards - values +
                           self.discount * tf.pad(values, [[0, 0], [0, 1]])[:, 1:])
                self.record("delta_v", tf.reduce_mean(delta_v))
                advantages = discounted_sum(delta_v, self.discount * self.lamb)
                self.record("advantages", tf.reduce_mean(advantages))

                # COMPUTE IMPORTANCE SAMPLING POLICY RATIO
                policy_ratio = tf.exp(self.policy.log_prob(actions, observations) -
                                      self.old_policy.log_prob(actions, observations))
                self.record("policy_ratio", tf.reduce_mean(policy_ratio))

                # COMPUTE CLIPPED SURROGATE POLICY LOSS
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
