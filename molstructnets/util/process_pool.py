import os
from multiprocessing import pool


default_number_processes = os.cpu_count()


class ProcessPool:

    def __init__(self, number_threads=default_number_processes):
        self.pool = pool.Pool(number_threads)
        self.number_threads = number_threads
        self.futures = []

    def submit(self, function_, *args, **kwargs):
        self.futures.append(self.pool.apply_async(function_, args, kwargs))
        return self.futures[-1]

    def wait(self):
        for i in range(len(self.futures)):
            self.futures[i].wait()
        self.futures = []

    def get_results(self):
        results = list()
        for i in range(len(self.futures)):
            results.append(self.futures[i].get())
        self.futures = []
        return results

    def get_number_threads(self):
        return self.number_threads

    def close(self):
        self.pool.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
