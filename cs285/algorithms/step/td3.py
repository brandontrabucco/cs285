"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.algorithms.step.step_algorithm import StepAlgorithm
import tensorflow as tf


class TD3(StepAlgorithm):

    def __init__(
        self,
        policy,
        target_policy,
        qf1,
        qf2,
        target_qf1,
        target_qf2,
        reward_scale=1.0,
        discount=0.99,
        tau=1e-2,
        target_noise=0.5,
        target_clipping=0.2,
        policy_delay=1,
        policy_optimizer_class=tf.keras.optimizers.Adam,
        policy_optimizer_kwargs=None,
        qf_optimizer_class=tf.keras.optimizers.Adam,
        qf_optimizer_kwargs=None,
        **kwargs,
    ):
        StepAlgorithm.__init__(self, **kwargs)
        self.policy = policy
        self.target_policy = target_policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2

        self.reward_scale = reward_scale
        self.discount = discount
        self.tau = tau
        self.target_noise = target_noise
        self.target_clipping = target_clipping
        self.policy_delay = policy_delay

        if policy_optimizer_kwargs is None:
            policy_optimizer_kwargs = dict(lr=0.0001, clipnorm=1.0)
        self.policy_optimizer = policy_optimizer_class(
            **policy_optimizer_kwargs)

        if qf_optimizer_kwargs is None:
            qf_optimizer_kwargs = dict(lr=0.001, clipnorm=1.0)
        self.qf_optimizer = qf_optimizer_class(**qf_optimizer_kwargs)

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
            mean_actions, log_pi = self.policy.expected_value(observations)
            next_mean_actions, next_log_pi = self.target_policy.expected_value(
                next_observations)

            # TARGET POLICY NOISE
            noise = tf.clip_by_value(
                self.target_noise * tf.random.normal(tf.shape(mean_actions)),
                -self.target_clipping, self.target_clipping)
            next_noisy_actions = next_mean_actions + noise

            # BUILD Q FUNCTION TARGET VALUE
            inputs = tf.concat([next_observations, next_noisy_actions], -1)
            target_qf1_value = self.target_qf1(inputs)
            self.record("target_qf1_value", tf.reduce_mean(target_qf1_value))
            target_qf2_value = self.target_qf2(inputs)
            self.record("target_qf2_value", tf.reduce_mean(target_qf2_value))
            qf_targets = tf.stop_gradient(
                self.reward_scale * rewards + terminals * self.discount * (
                    tf.minimum(target_qf1_value, target_qf2_value)))
            self.record("qf_targets", tf.reduce_mean(qf_targets))

            # BUILD Q FUNCTION LOSS
            inputs = tf.concat([observations, actions], -1)
            qf1_value = self.qf1(inputs)
            self.record("qf1_value", tf.reduce_mean(qf1_value))
            qf2_value = self.qf2(inputs)
            self.record("qf2_value", tf.reduce_mean(qf2_value))
            qf1_loss = tf.reduce_mean(tf.keras.losses.logcosh(qf_targets, qf1_value))
            self.record("qf1_loss", qf1_loss)
            qf2_loss = tf.reduce_mean(tf.keras.losses.logcosh(qf_targets, qf2_value))
            self.record("qf2_loss", qf2_loss)

            # BUILD POLICY LOSS
            inputs = tf.concat([observations, mean_actions], -1)
            policy_qf1_value = self.qf1(inputs)
            self.record("policy_qf1_value", tf.reduce_mean(policy_qf1_value))
            policy_qf2_value = self.qf2(inputs)
            self.record("policy_qf2_value", tf.reduce_mean(policy_qf2_value))
            policy_loss = -tf.reduce_mean(
                tf.minimum(policy_qf1_value, policy_qf2_value))
            self.record("policy_loss", policy_loss)

        # BACK PROP GRADIENTS
        qf1_gradients = tape.gradient(qf1_loss, self.qf1.trainable_variables)
        self.qf_optimizer.apply_gradients(zip(
            qf1_gradients, self.qf1.trainable_variables))

        qf2_gradients = tape.gradient(qf2_loss, self.qf2.trainable_variables)
        self.qf_optimizer.apply_gradients(zip(
            qf2_gradients, self.qf2.trainable_variables))

        # DELAYED POLICY UPDATE
        if self.inner_iteration % self.policy_delay == 0:
            policy_gradients = tape.gradient(policy_loss, self.policy.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(
                policy_gradients, self.policy.trainable_variables))

            # SOFT UPDATE TARGET PARAMETERS
            self.target_policy.set_weights([
                self.tau * w1 + (1.0 - self.tau) * w2 for w1, w2 in zip(
                    self.policy.get_weights(), self.target_policy.get_weights())])

            self.target_qf1.set_weights([
                self.tau * w1 + (1.0 - self.tau) * w2 for w1, w2 in zip(
                    self.qf1.get_weights(), self.target_qf1.get_weights())])

            self.target_qf2.set_weights([
                self.tau * w1 + (1.0 - self.tau) * w2 for w1, w2 in zip(
                    self.qf2.get_weights(), self.target_qf2.get_weights())])
