from util import progressbar, logger
import multiprocessing
import threading


class MultiProcessProgressbar():

    def __init__(self, max_value, log_level=logger.LogLevel.INFO, value_buffer=1):
        self._value_buffer = value_buffer
        if log_level >= logger.global_log_level:
            self._queue = multiprocessing.Manager().Queue()
            self._listener = threading.Thread(target=run_progressbar_listener, args=(max_value, log_level, self._queue))
            self._listener.start()
            self._value = 0
        else:
            self._queue = None

    def get_slave(self):
        return MultiProcessProgressbarSlave(self._queue, self._value_buffer)

    def increment(self, value=1):
        if self._queue is not None:
            self._value += value
            if self._value >= self._value_buffer:
                self._queue.put(self._value)
                self._value = 0

    def finish(self):
        if self._queue is not None:
            if self._value > 0:
                self._queue.put(self._value)
            self._queue.put(None)
            self._listener.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


class MultiProcessProgressbarSlave():

    def __init__(self, queue, value_buffer):
        self._queue = queue
        self._value_buffer = value_buffer
        self._value = 0

    def increment(self, value=1):
        if self._queue is not None:
            self._value += value
            if self._value >= self._value_buffer:
                self._queue.put(self._value)
                self._value = 0

    def finish(self):
        if self._queue is not None:
            if self._value > 0:
                self._queue.put(self._value)


def run_progressbar_listener(max_value, log_level, queue):
    with progressbar.ProgressBar(max_value, log_level) as progress:
        running = True
        while running:
            value = queue.get()
            if value is None:
                running = False
            else:
                progress.increment(value)
