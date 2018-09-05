import math
import queue

from steps.preprocessing.shared.tensor2d import tensor_2d_jit_array
from util import thread_pool, misc


class TrainingArrays():

    def __init__(self, global_parameters, epochs, batch_size, multi_process=True):
        self._array = tensor_2d_jit_array.load_array(global_parameters, train=True, transform=True,
                                                     multi_process=multi_process)
        preprocess_size = misc.max_in_memory_chunk_size(self._array.dtype, self._array.shape, use_swap=False,
                                                        fraction=1 / 3)
        preprocess_size -= preprocess_size % batch_size
        queue_size = math.ceil(preprocess_size / batch_size)
        preprocess_size = min(len(self._array), preprocess_size)
        input_queue = queue.Queue(queue_size)
        output_queue = queue.Queue(queue_size)
        self._input_array = QueueArray(self._array.shape, input_queue)
        self._output_array = QueueArray((len(self._array), 2), output_queue)
        self._pool = thread_pool.ThreadPool(1)
        self._pool.submit(preprocess_batches, self._array, epochs, batch_size, input_queue, output_queue,
                          preprocess_size)

    @property
    def input(self):
        return self._input_array

    @property
    def output(self):
        return self._output_array

    def close(self):
        self._pool.close()
        self._array.close()


class QueueArray():

    def __init__(self, shape, queue):
        self._shape = shape
        self._queue = queue

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


def preprocess_batches(array, epochs, batch_size, input_queue, output_queue, preprocess_size=None):
    if preprocess_size is None:
        preprocess_size = len(array)
    for epoch in range(epochs):
        preprocess_offset = 0
        while preprocess_offset < len(array):
            preprocess_next = preprocess_offset + min(preprocess_size, len(array) - preprocess_offset)
            data = array[preprocess_offset:preprocess_next]
            offset = 0
            while offset < len(data):
                next = offset + min(batch_size, len(data) - offset)
                input_queue.put(data[offset:next])
                output_queue.put(array.classes(slice(preprocess_offset + offset, preprocess_offset + next)))
                offset = next
            preprocess_offset = preprocess_next
        array.shuffle()
