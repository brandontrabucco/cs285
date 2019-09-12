"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from cs285.core.monitors.monitor import Monitor
from cs285.core.monitors.create_monitor_thread import create_monitor_thread
import os
import queue
import threading
import tensorflow as tf


class LocalMonitor(Monitor):

    def __init__(
        self,
        logging_dir
    ):
        # create a separate tensor board logging thread
        self.logging_dir = logging_dir

        # communicate with the thread via a queue
        self.record_queue = queue.Queue()

        # listen on the specified thread in a loop
        self.thread = threading.Thread(
            target=create_monitor_thread, args=(logging_dir, self.record_queue))
        self.thread.start()
        
        # keep track of the current global step, and resume training if necessary
        self.step_file = os.path.join(logging_dir, "step")
        if tf.io.gfile.exists(self.step_file):
            with tf.io.gfile.GFile(self.step_file, "r") as f:
                self.step = int(f.read().strip())

        # otherwise start from scratch
        else:
            self.step = 0
        self.lock = threading.Lock()

    def increment(
        self
    ):
        # increment how many steps have been collected so far
        self.lock.acquire()
        self.step += 1
        self.lock.release()

    def save_step(
        self
    ):
        # save the current step (like the tf global step) to a file
        self.lock.acquire()
        with tf.io.gfile.GFile(self.step_file, "w") as f:
            f.write(str(self.step))
        self.lock.release()

    def record(
        self,
        key,
        value,
    ):
        # record a tensor into the tensor board logging thread
        self.lock.acquire()
        self.record_queue.put((self.step, key, value))
        self.lock.release()
