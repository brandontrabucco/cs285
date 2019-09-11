"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.algorithms.step_algorithm import StepAlgorithm
import tensorflow as tf


class BehaviorCloning(StepAlgorithm):

    def __init__(
        self,
        policy,
        policy_optimizer_class=tf.keras.optimizers.Adam,
        policy_optimizer_kwargs=None,
        **kwargs,
    ):
        # train a policy using behavior cloning on expert data
        StepAlgorithm.__init__(self, **kwargs)
        self.policy = policy

        # build an optimizer for the policy
        if policy_optimizer_kwargs is None:
            policy_optimizer_kwargs = dict(lr=0.0001, clipnorm=1.0)
        self.policy_optimizer = policy_optimizer_class(
            **policy_optimizer_kwargs)

    def update_algorithm(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        terminals
    ):
        # update the policy with behavior cloning
        with tf.GradientTape() as tape:

            # compute the log probability of expert actions
            log_prob = self.policy.log_prob(actions, observations)
            self.record("log_prob", tf.reduce_mean(log_prob))

            # build the cross entropy loss
            policy_loss = -tf.reduce_mean(log_prob)
            self.record("policy_loss", policy_loss)

        # back prop gradients to maximize log prob of expert actions
        policy_gradients = tape.gradient(
            policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(
            policy_gradients, self.policy.trainable_variables))
