import hashlib


def hash_parameters(parameters):
    return hashlib.sha1(str(parameters).encode()).hexdigest()


def copy_dict_from_keys(dict_, keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = dict_[key]
    return new_dict


def in_range(value, min=None, max=None):
    if min is not None and value < min:
        return False
    if max is not None and value > max:
        return False
    return True
