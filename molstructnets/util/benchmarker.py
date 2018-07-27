import datetime
import time


class Benchmarker():

    def __init__(self):
        self._checkpoints = dict()
        self._order = list()
        self._last_checkpoint = time.time()

    def start(self):
        self._last_checkpoint = time.time()

    def checkpoint(self, name):
        if name not in self._checkpoints:
            self._order.append(name)
            self._checkpoints[name] = [0, 0]
        self._checkpoints[name][1] += 1
        current_time = time.time()
        self._checkpoints[name][0] += current_time - self._last_checkpoint
        self._last_checkpoint = current_time

    def print(self):
        for checkpoint_name in self._order:
            time_average = self._checkpoints[checkpoint_name][0] / self._checkpoints[checkpoint_name][1]
            print(checkpoint_name + ': ' + str(datetime.timedelta(seconds=time_average)))
