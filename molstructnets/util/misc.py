import hashlib


def hash_parameters(parameters):
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
