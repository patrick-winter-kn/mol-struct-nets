import progressbar
import threading


class MultithreadProgress:

    def __init__(self, max_value):
        self.progress = progressbar.ProgressBar(max_value=max_value)
        self.lock = threading.Lock()
        self.counter = 0

    def increment(self):
        self.lock.acquire()
        self.counter += 1
        self.progress.update(self.counter)
        self.lock.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.finish()
