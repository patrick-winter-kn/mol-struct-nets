import copy
import threading


class ConcurrentSet:

    def __init__(self):
        self.set = set()
        self.lock = threading.Lock()

    def add(self, value):
        self.lock.acquire()
        if value in self.set:
            result = False
        else:
            self.set.add(value)
            result = True
        self.lock.release()
        return result

    def contains(self, value):
        self.lock.acquire()
        result = value in self.set
        self.lock.release()
        return result

    def get_set_copy(self):
        self.lock.acquire()
        copied_set = copy.deepcopy(self.set)
        self.lock.release()
        return copied_set
