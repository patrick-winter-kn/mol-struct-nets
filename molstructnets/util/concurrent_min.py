import threading
import sys


class ConcurrentMin:

    def __init__(self):
        self.min = sys.maxsize
        self.lock = threading.Lock()

    def add_value(self, value):
        if value < self.min:
            self.lock.acquire()
            if value < self.min:
                self.min = value
            self.lock.release()

    def get_min(self):
        return self.min
