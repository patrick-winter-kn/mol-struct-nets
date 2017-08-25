import os
from concurrent import futures


number_threads = os.cpu_count() * 2


class ThreadPool:

    def __init__(self):
        self.pool = futures.ThreadPoolExecutor(number_threads)
        self.futures = []

    def submit(self, function, arguments):
        self.futures.append(self.pool.submit(function, arguments))

    def wait(self):
        for i in range(len(self.futures)):
            self.futures[i].result()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.shutdown()
