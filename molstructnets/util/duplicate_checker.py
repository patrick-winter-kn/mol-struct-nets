import threading


class DuplicateChecker:

    def __init__(self):
        self.set = set()
        self.lock = threading.Lock()

    def is_duplicate(self, value):
        self.lock.acquire()
        if value in self.set:
            result = True
        else:
            self.set.add(value)
            result = False
        self.lock.release()
        return result
