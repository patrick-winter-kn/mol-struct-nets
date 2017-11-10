import hashlib
import math
from util import reference_data_set
import numpy


def hash_parameters(parameters):
    parameters = sorted(parameters.items())
    return hashlib.sha1(str(parameters).encode()).hexdigest()


def copy_dict_from_keys(dict_, keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = dict_[key]
    return new_dict


def in_range(value, min_=None, max_=None):
    if min_ is not None and value < min_:
        return False
    if max_ is not None and value > max_:
        return False
    return True


def chunk(number, number_chunks):
    chunks = []
    chunk_size = math.ceil(number / number_chunks)
    for i in range(number_chunks):
        start = chunk_size * i
        end = min(start + chunk_size, number) - 1
        size = end - start + 1
        chunks.append({'size': size, 'start': start, 'end': end})
    return chunks


def is_active(probabilities):
    return probabilities[0] > probabilities[1]


def copy_into_memory(array, as_bool=False):
    if isinstance(array, numpy.ndarray):
        if as_bool:
            array = array.astype(bool)
        return array
    else:
        return copy_ndarray(array, as_bool)


def copy_ndarray(array, as_bool=False):
    if isinstance(array, reference_data_set.ReferenceDataSet):
        array_copy = numpy.zeros(array.shape)
        array_copy[:] = array[:]
        if as_bool:
            array_copy = array_copy.astype(bool)
        return array_copy
    else:
        if as_bool:
            return array.astype(bool)
        else:
            return numpy.copy(array)


def substring_cut_from_middle(string, slices):
    removed = 0
    for slice_ in slices:
        cut_start = slice_[0] - removed
        cut_end = slice_[1] - removed
        string = string[:cut_start] + string[cut_end:]
        removed += slice_[1] - slice_[0]
    return string


def minimum(value_1, value_2):
    if value_1 is None:
        return value_2
    if value_2 is None:
        return value_1
    return min(value_1, value_2)


def maximum(value_1, value_2):
    if value_1 is None:
        return value_2
    if value_2 is None:
        return value_1
    return max(value_1, value_2)
