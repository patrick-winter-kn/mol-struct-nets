import hashlib


def hash_parameters(parameters):
    return hashlib.sha1(str(parameters).encode()).hexdigest()


def copy_dict_from_keys(dict_, keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = dict_[key]
    return new_dict
