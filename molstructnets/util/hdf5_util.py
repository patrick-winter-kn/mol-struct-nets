import h5py
from util import file_util


def set_property(file, key, value):
    is_file = isinstance(file, h5py.File)
    if not is_file:
        file = h5py.File(file_util.resolve_path(file), 'r+')
    file.attrs[key] = value
    if not is_file:
        file.close()


def delete_property(file, key):
    is_file = isinstance(file, h5py.File)
    if not is_file:
        file = h5py.File(file_util.resolve_path(file), 'r+')
    if key in file.attrs:
        del file.attrs[key]
    if not is_file:
        file.close()


def get_property(file, key):
    is_file = isinstance(file, h5py.File)
    if not is_file:
        if not file_util.file_exists(file):
            return None
        file = h5py.File(file_util.resolve_path(file), 'r')
    if key in file.attrs:
        value = file.attrs[key]
    else:
        value = None
    if not is_file:
        file.close()
    return value


def has_data_set(file, name):
    is_file = isinstance(file, h5py.File)
    if not is_file:
        if not file_util.file_exists(file):
            return None
        file = h5py.File(file_util.resolve_path(file), 'r')
    value = name in file
    if not is_file:
        file.close()
    return value


def create_dataset(file, name, shape, dtype='f', chunks=True):
    return file.create_dataset(name, shape, dtype=dtype, chunks=chunks, compression='gzip')


def create_dataset_from_data(file, name, data, chunks=True):
    return file.create_dataset(name, data=data, chunks=chunks, compression='gzip')
