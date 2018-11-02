import queue
import numpy
import math

from steps.preprocessing.shared.tensor2d import tensor_2d_array
from util import thread_pool, misc, constants


class MultitargetTrainingArrays():

    def __init__(self, global_parameters, epochs, batch_size, frozen_runs, number_batches, multi_process=True):
        self._arrays = list()
        input_shapes = list()
        output_shapes = list()
        data_sets = global_parameters[constants.GlobalParameters.data_set]
        for i in range(len(data_sets)):
            global_params = global_parameters.copy()
            global_params[constants.GlobalParameters.data_set] = data_sets[i][0]
            global_params[constants.GlobalParameters.target] = data_sets[i][1]
            global_params[constants.GlobalParameters.partition_data] = global_parameters[constants.GlobalParameters.partition_data][i]
            array = tensor_2d_array.load_array(global_params, train=True, transform=True, multi_process=multi_process)
            self._arrays.append(array)
            input_shapes.append(array.shape)
            output_shapes.append((len(array), 2))
        preprocess_size = misc.max_in_memory_chunk_size(self._arrays[0].dtype, self._arrays[0].shape, use_swap=False,
                                                        fraction=1 / 3)
        preprocess_size = math.floor(preprocess_size/len(self._arrays))
        preprocess_size -= preprocess_size % batch_size
        queue_size = int(preprocess_size / batch_size) * len(self._arrays)
        max_length = None
        for i in range(len(self._arrays)):
            max_length = misc.maximum(max_length, len(self._arrays[i]))
        self._batches_per_epoch = math.ceil(max_length / batch_size)
        input_queue = queue.Queue(queue_size)
        output_queue = queue.Queue(queue_size)
        self._input_array = QueueArray(input_shapes, batch_size, input_queue)
        self._output_array = QueueArray(output_shapes, batch_size, output_queue)
        self._pool = thread_pool.ThreadPool(1)
        self._pool.submit(preprocess_batches, self._arrays, epochs, batch_size, frozen_runs, number_batches,
                          self._batches_per_epoch, input_queue, output_queue, preprocess_size)

    @property
    def input(self):
        return self._input_array

    @property
    def output(self):
        return self._output_array

    def batches_per_epoch(self):
        return self._batches_per_epoch

    def next_data_set(self):
        self._input_array.next_data_set()
        self._output_array.next_data_set()

    def close(self):
        self._pool.close()
        for array in self._arrays:
            array.close()


class QueueArray():

    def __init__(self, shapes, slice_size, queue):
        self._current_shape = None
        self._shapes = shapes
        self._shape_index = 0
        self._slice_start = list()
        for i in range(len(shapes)):
            self._slice_start.append(0)
        self._queue = queue
        self._slice_size = slice_size
        self.update_current_shape()

    @property
    def shape(self):
        return self._current_shape

    @property
    def ndim(self):
        return len(self._current_shape)

    def __len__(self):
        return self._current_shape[0]

    def __getitem__(self, item):
        return self._queue.get()

    def next_data_set(self):
        self._shape_index = (self._shape_index + 1) % len(self._shapes)
        self._slice_start[self._shape_index] = (self._slice_start[self._shape_index] + self._slice_size)\
                                               % self._shapes[self._shape_index][0]
        self.update_current_shape()

    def update_current_shape(self):
        self._current_shape = list(self._shapes[self._shape_index])
        self._current_shape[0] = min(self._slice_size, self._current_shape[0] - self._slice_start[self._shape_index])
        self._current_shape = tuple(self._current_shape)


def preprocess_batches(arrays, epochs, batch_size, frozen_runs, number_batches, batches_per_epoch, input_queue, output_queue, preprocess_size=None):
    if preprocess_size is None:
        preprocess_size = batch_size * batches_per_epoch
    dtype = arrays[0].dtype
    shape = list(arrays[0].shape)
    shape[0] = preprocess_size
    shape = tuple(shape)
    inputs = list()
    outputs = list()
    offsets = list()
    for i in range(len(arrays)):
        inputs.append(numpy.zeros(shape, dtype))
        outputs.append(numpy.zeros((preprocess_size, 2), 'float32'))
        offsets.append(0)
    for epoch in range(epochs):
        for i in range(math.ceil((batches_per_epoch * batch_size) / preprocess_size)):
            remaining_batches = batches_per_epoch - (i * int(preprocess_size/batch_size))
            # preprocessing
            for data_set_index in range(len(arrays)):
                done = 0
                array = arrays[data_set_index]
                todo = min(preprocess_size, remaining_batches * batch_size)
                while done < todo:
                    start = offsets[data_set_index]
                    end = min(len(array), start + todo - done)
                    size = end - start
                    inputs[data_set_index][done:done+size] = array[start:end]
                    outputs[data_set_index][done:done+size] = array.classes(slice(start, end))
                    done += size
                    offsets[data_set_index] += size
                    if offsets[data_set_index] == len(array):
                        offsets[data_set_index] = 0
                        array.shuffle()
            # filling queues
            for batch_number in range(min(int(preprocess_size/batch_size), remaining_batches)):
                offset = batch_number * batch_size
                for data_set_index in range(len(arrays)):
                    for j in range(frozen_runs + 1):
                        input_queue.put(inputs[data_set_index][offset:offset+batch_size])
                        output_queue.put(outputs[data_set_index][offset:offset+batch_size])
