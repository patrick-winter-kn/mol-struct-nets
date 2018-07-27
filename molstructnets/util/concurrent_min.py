import threading

from util import misc


class ConcurrentMin:

    def __init__(self):
        self.min = None
        self.lock = threading.Lock()

    def add(self, value):
        if value is not None and self.min is None or value < self.min:
            self.lock.acquire()
            self.min = misc.minimum(self.min, value)
            self.lock.release()

    def get_min(self):
        return self.min
