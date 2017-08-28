import os
from concurrent import futures


default_number_threads = os.cpu_count() * 2


class ThreadPool:

    def __init__(self, number_threads=default_number_threads):
        self.pool = futures.ThreadPoolExecutor(number_threads)
        self.number_threads = number_threads
        self.futures = []

    def submit(self, function, *args, **kwargs):
        self.futures.append(self.pool.submit(function, *args, **kwargs))

    def wait(self):
        for i in range(len(self.futures)):
            self.futures[i].result()

    def get_number_threads(self):
        return self.number_threads

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.shutdown()
