import os
from multiprocessing import pool
from util import logger
import traceback

default_number_processes = os.cpu_count()
open_process_pools = list()


class ProcessPool:

    def __init__(self, number_threads=default_number_processes):
        self.pool = pool.Pool(number_threads)
        self.number_threads = number_threads
        self.futures = []
        open_process_pools.append(self)

    def submit(self, function_, *args, **kwargs):
        args = tuple([function_] + list(args))
        self.futures.append(self.pool.apply_async(run_function, args, kwargs))
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
        open_process_pools.remove(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def close_all_pools():
    for process_pool in open_process_pools:
        process_pool.close()


def run_function(function_, *args, **kwargs):
    try:
        return function_(*args, **kwargs)
    except BaseException as e:
        logger.log('Error while running ' + function_.__name__ + '()', log_level=logger.LogLevel.ERROR)
        traceback.print_exc()
        raise e
