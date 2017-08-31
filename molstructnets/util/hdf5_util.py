import h5py
from util import file_util


def set_property(file_path, key, value):
    data_h5 = h5py.File(file_util.resolve_path(file_path), 'r+')
    data_h5.attrs[key] = value
    data_h5.close()


def get_property(file_path, key):
    data_h5 = h5py.File(file_util.resolve_path(file_path), 'r')
    if key in data_h5.attrs:
        value = data_h5.attrs[key]
    else:
        value = None
    data_h5.close()
    return value
