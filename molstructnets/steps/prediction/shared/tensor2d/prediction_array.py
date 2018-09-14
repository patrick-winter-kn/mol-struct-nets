import math
import queue

from steps.preprocessing.shared.tensor2d import tensor_2d_array
from util import thread_pool, misc


class PredictionArrays():

    def __init__(self, global_parameters, batch_size, test=False, runs=1, transformations=1, multi_process=True, percent=1):
        self._array = tensor_2d_array.load_array(global_parameters, test=test, transform=transformations > 1,
                                                 multi_process=multi_process, percent=percent)
        preprocess_size = misc.max_in_memory_chunk_size(self._array.dtype, self._array.shape, use_swap=False,
                                                        fraction=1 / 3)
        preprocess_size -= preprocess_size % batch_size
        queue_size = int(preprocess_size / batch_size)
        preprocess_size = min(len(self._array), preprocess_size)
        input_queue = queue.Queue(queue_size)
        self._input_array = QueueArray(self._array.shape, input_queue)
        self._pool = thread_pool.ThreadPool(1)
        self._pool.submit(preprocess_batches, self._array, batch_size, input_queue, runs, transformations, preprocess_size)

    @property
    def input(self):
        return self._input_array

    @property
    def output(self):
        return self._array.classes()

    def close(self):
        self._pool.close()
        self._array.close()


class QueueArray():

    def __init__(self, shape, queue_):
        self._shape = shape
        self._queue = queue_

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def __len__(self):
        return self._shape[0]

    def __getitem__(self, item):
        return self._queue.get()

    def next(self):
        return self._queue.get()


def preprocess_batches(array, batch_size, input_queue, runs=1, transformations=1, preprocess_size=None):
    if preprocess_size is None:
        preprocess_size = len(array)
    for run in range(runs):
        for i in range(transformations):
            array.set_iteration(i)
            preprocess_offset = 0
            while preprocess_offset < len(array):
                preprocess_next = preprocess_offset + min(preprocess_size, len(array) - preprocess_offset)
                data = array[preprocess_offset:preprocess_next]
                offset = 0
                while offset < len(data):
                    next = offset + min(batch_size, len(data) - offset)
                    input_queue.put(data[offset:next])
                    offset = next
                preprocess_offset = preprocess_next
