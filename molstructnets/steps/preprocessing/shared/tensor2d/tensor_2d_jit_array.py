import queue
import random

import h5py
import numpy
from numpy import random

from steps.preprocessing.shared.tensor2d import tensor_2d_jit_preprocessor
from util import file_structure, constants, process_pool, misc, buffered_queue


class Tensor2DJitArray():

    def __init__(self, smiles, classes, indices, preprocessed_path, random_seed, multi_process=True):
        self._smiles = smiles
        self._classes = classes
        self._indices = indices
        if isinstance(multi_process, process_pool.ProcessPool):
            self._pool = multi_process
        elif multi_process:
            self._pool = process_pool.ProcessPool()
        else:
            self._pool = None
        self._preprocessor = tensor_2d_jit_preprocessor.Tensor2DJitPreprocessor(preprocessed_path)
        self._shape = tuple([len(self._indices)] + list(self._preprocessor.shape))
        self._random_seed = random_seed
        self._iteration = 0

    def shuffle(self):
        random.shuffle(self._indices)
        self._iteration += 1

    def set_iteration(self, iteration):
        self._iteration = iteration

    def __len__(self):
        return self._shape[0]

    def __getitem__(self, item):
        indices = self._indices[item]
        single_item = False
        if not hasattr(indices, '__len__'):
            single_item = True
            indices = [indices]
        random_seed = None
        if self._pool is not None and len(indices) > 1:
            all_results = numpy.zeros([len(indices)] + list(self._preprocessor.shape), dtype='float32')
            chunks = misc.chunk(len(indices), self._pool.get_number_threads())
            data_queue = buffered_queue.BufferedQueue(10)
            for chunk in chunks:
                indices_chunk = indices[chunk['start']:chunk['end']]
                if self._random_seed is not None:
                    random_seed = self._random_seed + chunk['start'] + self._iteration * len(self)
                self._pool.submit(self._preprocessor.preprocess, self._smiles[indices_chunk], chunk['start'],
                                  data_queue, random_seed)
            done = 0
            while done < len(indices):
                preprocessed = data_queue.get()
                preprocessed.fill_array(all_results)
                done += 1
            return all_results
        else:
            if self._random_seed is not None:
                random_seed = self._random_seed + indices[0] + self._iteration * len(self)
            data_queue = queue.Queue()
            self._preprocessor.preprocess(self._smiles[indices], 0, data_queue, random_seed)
            result = numpy.zeros([len(indices)] + list(self._preprocessor.shape), dtype='float32')
            preprocessed = data_queue.get()
            preprocessed.fill_array(result)
            if single_item:
                result = result[0]
            return result

    def calc_substructure_locations(self, start, end, substructures, location_queue, only_substructures=False,
                                    only_atoms=False):
        random_seed = None
        if self._pool is not None and len(self._smiles) > 1:
            chunks = misc.chunk(end - start, self._pool.get_number_threads())
            for chunk in chunks:
                indices_chunk = range(start + chunk['start'], start + chunk['end'])
                if self._random_seed is not None:
                    random_seed = self._random_seed + start + chunk['start'] + self._iteration * len(self)
                self._pool.submit(self._preprocessor.substructure_locations, self._smiles[indices_chunk], substructures,
                                  chunk['start'], location_queue, random_seed, only_substructures, only_atoms)

    def calc_atom_locations(self, start, end, location_queue):
        random_seed = None
        if self._pool is not None and len(self._smiles) > 1:
            chunks = misc.chunk(end - start, self._pool.get_number_threads())
            for chunk in chunks:
                indices_chunk = range(start + chunk['start'], start + chunk['end'])
                if self._random_seed is not None:
                    random_seed = self._random_seed + start + chunk['start'] + self._iteration * len(self)
                self._pool.submit(self._preprocessor.atom_locations, self._smiles[indices_chunk], chunk['start'],
                                  location_queue, random_seed)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return numpy.float32

    def smiles(self, item=None):
        if item is None:
            item = slice(0, len(self._indices))
        return self._smiles[self._indices[item]]

    def classes(self, item=None):
        if item is None:
            item = slice(0, len(self._indices))
        return self._classes[self._indices[item]]

    def close(self):
        if self._pool is not None:
            self._pool.close()


def load_array(global_parameters, train=False, test=False, transform=False, multi_process=True):
    smiles_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
    smiles = smiles_h5[file_structure.DataSet.smiles][:]
    smiles_h5.close()
    classes_h5 = h5py.File(file_structure.get_target_file(global_parameters), 'r')
    classes = classes_h5[file_structure.Target.classes][:]
    classes_h5.close()
    if train:
        partition_h5 = h5py.File(file_structure.get_partition_file(global_parameters), 'r')
        partition = partition_h5[file_structure.Partitions.train][:]
        partition_h5.close()
    elif test:
        partition_h5 = h5py.File(file_structure.get_partition_file(global_parameters), 'r')
        partition = partition_h5[file_structure.Partitions.test][:]
        partition_h5.close()
    else:
        partition = numpy.arange(len(smiles), dtype='uint32')
    if transform:
        random_seed = global_parameters[constants.GlobalParameters.seed]
    else:
        random_seed = None
    preprocessed_path = global_parameters[constants.GlobalParameters.preprocessed_data]
    return Tensor2DJitArray(smiles, classes, partition, preprocessed_path, random_seed, multi_process=multi_process)
