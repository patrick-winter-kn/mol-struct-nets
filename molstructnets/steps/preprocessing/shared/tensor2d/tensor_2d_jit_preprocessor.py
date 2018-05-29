import random
import numpy
import h5py
from util import hdf5_util, file_structure


class Tensor2DJitPreprocessor:

    def __init__(self, preprocessed_path):
        self._shape = tuple(hdf5_util.get_property(preprocessed_path, file_structure.PreprocessedTensor2DJit.dimensions))
        # TODO
        pass

    def preprocess(self, smiles_array, random_seed):
        results = numpy.zeros([len(smiles_array)] + list(self.shape), dtype='float32')
        for i in range(len(smiles_array)):
            smiles = smiles_array[i].decode('utf-8')
            # TODO
        return results

    @property
    def shape(self):
        return self._shape
