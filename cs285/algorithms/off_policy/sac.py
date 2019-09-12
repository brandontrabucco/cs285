"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.algorithms.algorithm import Algorithm
import tensorflow as tf
import math


class SAC(Algorithm):

    def __init__(
        self,
        policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        reward_scale=1.0,
        discount=0.99,
        tau=0.005,
        policy_delay=1,
        target_entropy=(-1.0),
        initial_alpha=0.01,
        qf_optimizer_class=tf.keras.optimizers.Adam,
        qf_optimizer_kwargs=None,
        policy_optimizer_class=tf.keras.optimizers.Adam,
        policy_optimizer_kwargs=None,
        alpha_optimizer_class=tf.keras.optimizers.Adam,
        alpha_optimizer_kwargs=None,
        **kwargs,
    ):
        # train a policy using twin soft actor critic
        Algorithm.__init__(self, **kwargs)
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        # control the scale and decay of rewards
        self.reward_scale = reward_scale
        self.discount = discount

        # control the parameters of sac
        self.tau = tau
        self.policy_delay = policy_delay
        self.target_entropy = target_entropy
        self.log_alpha = tf.Variable(math.log(abs(initial_alpha)), dtype=tf.float32)

        # build an optimizer for the q function
        if qf_optimizer_kwargs is None:
            qf_optimizer_kwargs = dict(lr=0.001, clipnorm=1.0)
        self.qf_optimizer = qf_optimizer_class(**qf_optimizer_kwargs)

        # build an optimizer for the policy
        if policy_optimizer_kwargs is None:
            policy_optimizer_kwargs = dict(lr=0.0001, clipnorm=1.0)
        self.policy_optimizer = policy_optimizer_class(
            **policy_optimizer_kwargs)

        # build an optimizer for the alpha factor
        if alpha_optimizer_kwargs is None:
            alpha_optimizer_kwargs = dict(lr=0.0001, clipnorm=1.0)
        self.alpha_optimizer = alpha_optimizer_class(
            **alpha_optimizer_kwargs)

        # used to track when to update the target networks
        self.inner_iteration = 0

    def update_algorithm(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        terminals
    ):
        self.inner_iteration += 1
        with tf.GradientTape(persistent=True) as tape:
            alpha = tf.exp(self.log_alpha)
            self.record("alpha", tf.reduce_mean(alpha))

            # SAMPLE ACTIONS FROM CURRENT POLICY
            sampled_actions, log_pi = self.policy.sample(observations)
            self.record("entropy", tf.reduce_mean(-log_pi))
            next_sampled_actions, next_log_pi = self.policy.sample(next_observations)
            self.record("next_entropy", tf.reduce_mean(-next_log_pi))

            # BUILD Q FUNCTION TARGET VALUE
            inputs = tf.concat([next_observations, next_sampled_actions], -1)
            target_qf1_value = self.target_qf1(inputs)[..., 0]
            self.record("target_qf1_value", tf.reduce_mean(target_qf1_value))
            target_qf2_value = self.target_qf2(inputs)[..., 0]
            self.record("target_qf2_value", tf.reduce_mean(target_qf2_value))
            qf_targets = tf.stop_gradient(
                self.reward_scale * rewards + terminals * self.discount * (
                    tf.minimum(target_qf1_value, target_qf2_value) - alpha * next_log_pi))
            self.record("qf_targets", tf.reduce_mean(qf_targets))

            # BUILD Q FUNCTION LOSS
            inputs = tf.concat([observations, actions], -1)
            qf1_value = self.qf1(inputs)[..., 0]
            self.record("qf1_value", tf.reduce_mean(qf1_value))
            qf2_value = self.qf2(inputs)[..., 0]
            self.record("qf2_value", tf.reduce_mean(qf2_value))
            qf1_loss = tf.reduce_mean(tf.keras.losses.logcosh(qf_targets, qf1_value))
            self.record("qf1_loss", qf1_loss)
            qf2_loss = tf.reduce_mean(tf.keras.losses.logcosh(qf_targets, qf2_value))
            self.record("qf2_loss", qf2_loss)

            # BUILD POLICY LOSS
            inputs = tf.concat([observations, sampled_actions], -1)
            policy_qf1_value = self.qf1(inputs)[..., 0]
            self.record("policy_qf1_value", tf.reduce_mean(policy_qf1_value))
            policy_qf2_value = self.qf2(inputs)[..., 0]
            self.record("policy_qf2_value", tf.reduce_mean(policy_qf2_value))
            policy_loss = tf.reduce_mean(alpha * log_pi - tf.minimum(
                policy_qf1_value, policy_qf2_value))
            self.record("policy_loss", policy_loss)

            alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(
                log_pi + self.target_entropy))
            self.record("alpha_loss", alpha_loss)

        # BACK PROP GRADIENTS
        qf1_gradients = tape.gradient(qf1_loss, self.qf1.trainable_variables)
        self.qf_optimizer.apply_gradients(zip(
            qf1_gradients, self.qf1.trainable_variables))

        qf2_gradients = tape.gradient(qf2_loss, self.qf2.trainable_variables)
        self.qf_optimizer.apply_gradients(zip(
            qf2_gradients, self.qf2.trainable_variables))

        # DELAYED POLICY UPDATE
        if self.inner_iteration % self.policy_delay == 0:
            alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(
                alpha_gradients, [self.log_alpha]))

            policy_gradients = tape.gradient(policy_loss, self.policy.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(
                policy_gradients, self.policy.trainable_variables))

            # SOFT UPDATE TARGET PARAMETERS
            self.target_qf1.set_weights([
                self.tau * w1 + (1.0 - self.tau) * w2 for w1, w2 in zip(
                    self.qf1.get_weights(), self.target_qf1.get_weights())])

            self.target_qf2.set_weights([
                self.tau * w1 + (1.0 - self.tau) * w2 for w1, w2 in zip(
                    self.qf2.get_weights(), self.target_qf2.get_weights())])
