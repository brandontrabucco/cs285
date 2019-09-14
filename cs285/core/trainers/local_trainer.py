"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.core.trainers.trainer import Trainer
import time


class LocalTrainer(Trainer):

    def __init__(
        self,
        warm_up_sampler,
        explore_sampler,
        eval_sampler,
        buffer,
        algorithm,
        num_epochs=1000,
        num_episodes_per_epoch=1,
        num_trains_per_epoch=1,
        num_episodes_before_train=10,
        num_epochs_per_eval=10,
        num_episodes_per_eval=10,
        saver=None,
        monitor=None
    ):
        # samplers are for collecting data
        self.warm_up_sampler = warm_up_sampler
        self.explore_sampler = explore_sampler
        self.eval_sampler = eval_sampler

        # buffers are for storing samplers
        self.buffer = buffer

        # algorithms are trained using the collected samplers
        self.algorithm = algorithm

        # specify how to train the policy
        self.num_epochs = num_epochs
        self.num_episodes_per_epoch = num_episodes_per_epoch
        self.num_trains_per_epoch = num_trains_per_epoch
        self.num_episodes_before_train = num_episodes_before_train

        # specify how to evaluate the policy
        self.num_epochs_per_eval = num_epochs_per_eval
        self.num_episodes_per_eval = num_episodes_per_eval

        # specify what to save and what to log about the model
        self.saver = saver
        self.monitor = monitor
        self.start_time = time.time()

    def train(
        self
    ):
        # train the model and load pre trained networks
        if self.saver is not None:
            self.saver.load()

        # warm up the buffer by collecting initial samples
        warm_up_paths, warm_up_return, warm_up_steps = self.warm_up_sampler.collect(
            self.num_episodes_before_train, evaluate=False)

        # insert these samples into the buffer
        for path in warm_up_paths:
            self.buffer.insert_path(
                path["observations"], path["actions"], path["rewards"])

        # train for num_epochs rounds
        for epoch in range(self.num_epochs):
            # collect paths every round for exploration
            start_collect_time = time.time()
            explore_paths, explore_return, explore_steps = self.explore_sampler.collect(
                self.num_episodes_per_epoch, evaluate=False)

            # insert the exploration paths into the buffer
            for path in explore_paths:
                self.buffer.insert_path(
                    path["observations"], path["actions"], path["rewards"])

            # record the rate of step collection for logging purposes
            if self.monitor is not None:
                self.monitor.record(
                    "steps_per_second",
                    explore_steps / (time.time() - start_collect_time))

            # evaluate the policy sometimes
            if epoch % self.num_epochs_per_eval == 0:
                eval_paths, eval_return, eval_steps = self.eval_sampler.collect(
                    self.num_episodes_per_eval, evaluate=True)

                # record the mean return of the policy
                if self.monitor is not None:
                    self.monitor.record("eval_mean_return", eval_return)
                print("[{0:09d} / {1:09d}]  eval average return: {2}".format(
                    epoch, self.num_epochs, eval_return))

                # save the policy after evaluating it and save the iteration as well
                if self.saver is not None:
                    self.saver.save()
                    if self.monitor is not None:
                        self.monitor.save_step()

            # train the model for num_trains_per_epoch steps per round
            for _i in range(self.num_trains_per_epoch if
                            self.num_trains_per_epoch > 0 else explore_steps):
                self.algorithm.fit(self.buffer)
                self.monitor.increment()

            # finally record how fast a round was completed for logging purposes
            if self.monitor is not None:
                self.monitor.record(
                    "epochs_per_second", epoch / (time.time() - self.start_time))
