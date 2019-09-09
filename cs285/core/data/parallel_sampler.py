"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.core.data.sampler import Sampler
import numpy as np
import threading


def collect_backend(
        inner_paths,
        inner_mean_returns,
        inner_steps_collected,
        inner_num_episodes,
        inner_evaluate,
        inner_render,
        inner_render_kwargs,
        inner_sampler
):
    # only collect if work is given
    if inner_num_episodes > 0:
        result_paths, result_mean_return, result_steps_collected = inner_sampler.collect(
            inner_num_episodes,
            evaluate=inner_evaluate,
            render=inner_render,
            **inner_render_kwargs)

        # push collected data into the main sampler thread
        inner_paths.extend(result_paths)
        inner_mean_returns.append(result_mean_return)
        inner_steps_collected.append(result_steps_collected)


class ParallelSampler(object):

    def __init__(
            self,
            *args,
            num_threads=1,
            **kwargs
    ):
        self.samplers = [Sampler(*args, **kwargs) for i in range(num_threads)]
        self.num_threads = num_threads

    def collect(
            self,
            num_episodes,
            evaluate=False,
            render=False,
            **render_kwargs
    ):
        # only spawn threads if paths need to be collected
        if num_episodes == 0:
            return [], 0.0, 0

        # collect many paths in parallel
        paths = []
        mean_returns = []
        steps_collected = []

        # start several sampler threads in parallel
        threads = [threading.Thread(
            target=collect_backend, args=(
                paths,
                mean_returns,
                steps_collected,
                # the first thread may have extension episodes to collect
                num_episodes // self.num_threads + (num_episodes % self.num_threads if i == 0 else 0),
                evaluate,
                render,
                render_kwargs,
                self.samplers[i])) for i in range(self.num_threads)]

        # wait until all samplers finish
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # merge the statistics from every sampler in the main thread
        return paths, np.mean(mean_returns, dtype=np.float32), np.sum(steps_collected, dtype=np.int32)
