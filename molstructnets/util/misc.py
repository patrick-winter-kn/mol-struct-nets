import hashlib
import math
import time

import humanize
import numpy
import psutil

from util import progressbar, logger


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
    number_chunks = math.ceil(number / chunk_size)
    for i in range(number_chunks):
        start = chunk_size * i
        end = min(start + chunk_size, number)
        size = end - start
        chunks.append({'size': size, 'start': start, 'end': end})
    return chunks


def chunk_by_size(number, max_chunk_size):
    chunks = []
    number_chunks = math.ceil(number / max_chunk_size)
    for i in range(number_chunks):
        start = max_chunk_size * i
        end = min(start + max_chunk_size, number)
        size = end - start
        chunks.append({'size': size, 'start': start, 'end': end})
    return chunks


def max_in_memory_chunk_size(dtype, shape, use_swap=True, fraction=1, buffer=2 * math.pow(1024, 3)):
    target_type = dtype
    shape = list(shape)
    shape[0] = 1
    shape = tuple(shape)
    single_size = numpy.zeros(shape, target_type).nbytes
    available_memory = psutil.virtual_memory().available
    if use_swap:
        available_memory += psutil.swap_memory().free
    available_memory -= buffer
    available_memory *= fraction
    if available_memory < single_size:
        return 0
    else:
        return math.floor(available_memory / single_size)


def minimum(value_1, value_2):
    if value_1 is None or numpy.isnan(value_1):
        return value_2
    if value_2 is None or numpy.isnan(value_2):
        return value_1
    return min(value_1, value_2)


def maximum(value_1, value_2):
    if value_1 is None or numpy.isnan(value_1):
        return value_2
    if value_2 is None or numpy.isnan(value_2):
        return value_1
    return max(value_1, value_2)


def keys_in(keys, dict_):
    contained = True
    for key in keys:
        contained &= key in dict_
    return contained


def to_int(string):
    try:
        return int(string)
    except Exception:
        return None


def normalize(values):
    values -= values.min()
    values /= values.max()


def be_busy(seconds):
    target = time.time() + seconds
    while time.time() < target:
        dummy = 0
        while dummy < 1000:
            dummy += 1
