import threading
import copy


class ConcurrentCountingSet:

    def __init__(self):
        self.dict = {}
        self.lock = threading.Lock()

    def add(self, value):
        self.lock.acquire()
        if value not in self.dict:
            self.dict[value] = 1
            result = True
        else:
            self.dict[value] += 1
            result = False
        self.lock.release()
        return result

    def update(self, values):
        self.lock.acquire()
        for value in values:
            if value not in self.dict:
                self.dict[value] = 1
            else:
                self.dict[value] += 1
        self.lock.release()

    def contains(self, value):
        self.lock.acquire()
        result = value in self.dict
        self.lock.release()
        return result

    def get_dict_copy(self):
        self.lock.acquire()
        copied_dict = copy.deepcopy(self.dict)
        self.lock.release()
        return copied_dict
