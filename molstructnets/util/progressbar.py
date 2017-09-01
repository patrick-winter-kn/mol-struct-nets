import progressbar
import threading
from util import logger


class ProgressBar:

    def __init__(self, max_value, log_level=logger.LogLevel.INFO):
        if logger.do_print(log_level):
            self.progress = progressbar.ProgressBar(max_value=max_value)
        else:
            self.progress = None
        self.lock = threading.Lock()
        self.counter = 0

    def increment(self, value=1):
        self.lock.acquire()
        self.counter += value
        if self.progress is not None:
            self.progress.update(self.counter)
        self.lock.release()

    def finish(self):
        if self.progress is not None:
            self.progress.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
