"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.baselines.baseline import Baseline
from cs285.distributions.continuous.gaussian import Gaussian
from cs285.data.samplers.parallel_sampler import ParallelSampler
import tensorflow as tf
import threading
import numpy as np


class EvaluatePolicy(Baseline):

    def __init__(
            self,
            *args,
            policy_ckpt="./policy.ckpt",
            num_episodes=10,
            num_threads=10,
            max_path_length=1000,
            policy_distribution_class=Gaussian,
            policy_distribution_kwargs=None,
            **kwargs
    ):
        self.all_policy_ckpts = tf.io.gfile.glob(policy_ckpt)
        self.num_episodes = num_episodes
        self.num_threads = num_threads
        self.max_path_length = max_path_length
        self.policy_distribution_class = policy_distribution_class
        if policy_distribution_kwargs is None:
            policy_distribution_kwargs = dict(std=0.1)
        self.policy_distribution_kwargs = policy_distribution_kwargs

        Baseline.__init__(
            self,
            *args,
            **kwargs)

    def evaluation_thread(
            self,
            policy_ckpt,
            mean_return_list
    ):
        policy = self.policy_distribution_class(
            tf.keras.models.load_model(policy_ckpt, compile=False),
            **self.policy_distribution_kwargs)
        paths, mean_return, steps = ParallelSampler(
            self.get_env,
            policy,
            num_threads=self.num_threads,
            max_path_length=self.max_path_length,
            selector=self.selector).collect(self.num_episodes, evaluate=True)

        mean_return_list.append(mean_return)

    def launch(
            self
    ):
        results = []
        threads = []
        for policy_ckpt in self.all_policy_ckpts:
            threads.append(
                threading.Thread(
                    target=self.evaluation_thread, args=(policy_ckpt, results)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return np.mean(results), np.std(results)
