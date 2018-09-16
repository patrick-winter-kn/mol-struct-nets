from concurrent import futures

from util import process_pool

default_number_threads = process_pool.default_number_processes


class ThreadPool:

    def __init__(self, number_threads=default_number_threads):
        self.pool = futures.ThreadPoolExecutor(number_threads)
        self.number_threads = number_threads
        self.futures = []

    def submit(self, function_, *args, **kwargs):
        args = tuple([function_] + list(args))
        self.futures.append(self.pool.submit(process_pool.run_function, *args, **kwargs))
        return self.futures[-1]

    def wait(self):
        for i in range(len(self.futures)):
            self.futures[i].result()
        self.futures = []

    def get_results(self):
        results = []
        for i in range(len(self.futures)):
            results.append(self.futures[i].result())
        self.futures = []
        return results

    def get_number_threads(self):
        return self.number_threads

    def close(self):
        self.pool.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
