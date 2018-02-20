import threading
from util import misc


class ConcurrentMax:

    def __init__(self):
        self.max = None
        self.lock = threading.Lock()

    def add(self, value):
        if value is not None and self.max is None or value > self.max:
            self.lock.acquire()
            self.max = misc.maximum(self.max, value)
            self.lock.release()

    def get_max(self):
        return self.max
