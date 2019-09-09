"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import numpy as np


class Sampler(object):

    def __init__(
            self,
            make_env,
            make_policy,
            master_policy,
            max_path_length=1000,
            selector=(lambda x: x),
            monitor=None
    ):
        self.env = make_env()
        self.worker_policy = make_policy()
        self.master_policy = master_policy
        self.max_path_length = max_path_length
        self.selector = selector
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
            current_observation = self.env.reset()
            path = {"observations": [], "actions": [], "rewards": []}
            path_return = 0.0

            # unroll the episode until done or max_path_length is attained
            for i in range(self.max_path_length):

                # select the correct input for the policy (if obs is not a tensor)
                inputs = self.selector(current_observation)[np.newaxis, ...]

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
                path["observations"].append(current_observation)
                path["actions"].append(action)
                path["rewards"].append(reward)
                current_observation = next_observation

                # decide to count the collected step towards the running total
                if not evaluate and self.monitor is not None:
                    self.monitor.increment()

                # exit if the simulation has reached a terminal state
                if done:
                    break

            # keep track of the transitions and returns across all episodes
            paths.append(path)
            mean_return += path_return / num_episodes

        # keep track of the total number of steps actually collected this session
        num_steps_collected = sum([len(p["rewards"]) for p in paths])
        return paths, mean_return, num_steps_collected
