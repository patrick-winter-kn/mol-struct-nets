from util import file_util, file_structure
import h5py
from keras import models
# TODO add file path to error messages


def validate_data_set(global_parameters):
    data_set_path = file_structure.get_data_set_file(global_parameters)
    error_prefix = 'Error while checking data set in ' + data_set_path + ':\n'
    if not file_util.file_exists(data_set_path):
        raise ValueError(error_prefix + 'File does not exist')
    with h5py.File(data_set_path, 'r') as data_set_h5:
        if 'smiles' not in data_set_h5.keys():
            raise ValueError(error_prefix + 'File does not contain \'smiles\' data set')


def validate_target(global_parameters):
    target_path = file_structure.get_target_file(global_parameters)
    error_prefix = 'Error while checking target in ' + target_path + ':\n'
    if not file_util.file_exists(target_path):
        raise ValueError(error_prefix + 'File does not exist')
    with h5py.File(target_path, 'r') as target_h5:
        if 'classes' not in target_h5.keys():
            raise ValueError(error_prefix + 'File does not contain \'classes\' data set')


def validate_partition(global_parameters):
    partition_path = global_parameters['partition_data']
    error_prefix = 'Error while checking partitions in ' + partition_path + ':\n'
    if not file_util.file_exists(partition_path):
        raise ValueError(error_prefix + 'File does not exist')
    with h5py.File(partition_path, 'r') as partition_h5:
        if 'train' not in partition_h5.keys():
            raise ValueError(error_prefix + 'File does not contain \'train\' data set')
        if 'test' not in partition_h5.keys():
            raise ValueError(error_prefix + 'File does not contain \'test\' data set')


def validate_preprocessed(global_parameters):
    preprocessed_path = global_parameters['preprocessed_data']
    error_prefix = 'Error while checking preprocessed data in ' + preprocessed_path + ':\n'
    if not file_util.file_exists(preprocessed_path):
        raise ValueError(error_prefix + 'File does not exist')
    with h5py.File(preprocessed_path, 'r') as preprocessed_h5:
        if 'preprocessed' not in preprocessed_h5.keys():
            raise ValueError(error_prefix + 'File does not contain \'preprocessed\' data set')


def validate_network(global_parameters):
    network_path = file_structure.get_network_file(global_parameters)
    error_prefix = 'Error while checking network in ' + network_path + ':\n'
    if not file_util.file_exists(network_path):
        raise ValueError(error_prefix + 'File does not exist')
    if models.load_model(network_path) is None:
        raise ValueError(error_prefix + 'Could not load network')


def validate_preprocessed_images(global_parameters):
    if not file_util.file_exists(global_parameters['preprocessed_data']):
        raise ValueError('Folder with preprocessed data does not exist')
    nr_files = len(file_util.list_files(global_parameters['preprocessed_data']))
    n = global_parameters['n']
    if nr_files < n:
        raise ValueError('Folder with preprocessed data contains less files than expected (' + str(nr_files) + '<' +
                         str(n) + ')')


def validate_prediction(global_parameters):
    prediction_path = file_structure.get_prediction_file(global_parameters)
    error_prefix = 'Error while prediction data in ' + prediction_path + ':\n'
    if not file_util.file_exists(prediction_path):
        raise ValueError(error_prefix + 'File does not exist')
    with h5py.File(prediction_path, 'r') as prediction_h5:
        if 'train' not in prediction_h5.keys():
            raise ValueError(error_prefix + 'File does not contain \'train\' data set')
        if 'test' not in prediction_h5.keys():
            raise ValueError(error_prefix + 'File does not contain \'test\' data set')
