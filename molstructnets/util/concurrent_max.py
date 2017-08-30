import threading


class ConcurrentMax:

    def __init__(self):
        self.max = 0
        self.lock = threading.Lock()

    def add_value(self, value):
        if value > self.max:
            self.lock.acquire()
            if value > self.max:
                self.max = value
            self.lock.release()

    def get_max(self):
        return self.max
