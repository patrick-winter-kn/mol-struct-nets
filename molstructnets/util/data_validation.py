from util import file_util, file_structure
import h5py
# TODO add file path to error messages


def validate_data_set(global_parameters):
    data_set_path = file_structure.get_data_set_file(global_parameters)
    if not file_util.file_exists(data_set_path):
        raise ValueError('Data set does not exist')
    with h5py.File(data_set_path, 'r') as data_set_h5:
        if 'smiles' not in data_set_h5.keys():
            raise ValueError('Data set does not contain SMILES data')


def validate_target(global_parameters):
    target_path = file_structure.get_target_file(global_parameters)
    if not file_util.file_exists(target_path):
        raise ValueError('Target does not exist')
    with h5py.File(target_path, 'r') as target_h5:
        if 'classes' not in target_h5.keys():
            raise ValueError('Target does not contain class probabilities data')
