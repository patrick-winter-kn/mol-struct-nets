import hashlib
import math


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
