"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.data.samplers.sampler import Sampler
import numpy as np


class SimpleSampler(Sampler):

    def __init__(
            self,
            make_env,
            master_policy,
            max_path_length=1000,
            selector=None,
            monitor=None
    ):
        self.env = make_env()
        self.worker_policy = master_policy.clone()
        self.master_policy = master_policy
        self.max_path_length = max_path_length
        self.selector = selector if selector is not None else (lambda x: x)
        self.monitor = monitor

    def collect(
            self,
            num_episodes,
            evaluate=False,
            render=False,
            **render_kwargs
    ):
        # copy parameters from the master policy to the worker
        self.worker_policy.set_weights(self.master_policy.get_weights())

        # keep track of the path variables and returns
        paths = []
        mean_return = 0.0

        # collect num_episodes amount of paths and track various things
        for episode in range(num_episodes):
            observation = self.env.reset()
            path_return = 0.0
            path = {"observations": [], "actions": [], "rewards": []}

            # unroll the episode until done or max_path_length is attained
            for i in range(self.max_path_length):

                # select the correct input for the policy (if obs is not a tensor)
                inputs = self.selector(observation)[np.newaxis, ...]

                # choose to use the stochastic or deterministic policy
                if evaluate:
                    action = self.worker_policy.expected_value(inputs)[0][0, ...]
                else:
                    action = self.worker_policy.sample(inputs)[0][0, ...]

                # update the environment
                next_observation, reward, done, info = self.env.step(action)
                path_return += reward

                # and possibly render the updated environment (to a video)
                if render:
                    self.env.render(**render_kwargs)

                # collect samples into the ongoing path lists
                path["observations"].append(observation)
                path["actions"].append(action)
                path["rewards"].append(reward)
                observation = next_observation

                # exit if the simulation has reached a terminal state
                if done:
                    break

            # keep track of the transitions and returns across all episodes
            paths.append(path)
            mean_return += path_return / num_episodes

        # keep track of the total number of steps actually collected this session
        num_steps_collected = sum([len(p["rewards"]) for p in paths])
        return paths, mean_return, num_steps_collected
