from util import file_util, file_structure
import h5py
from keras import models
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


def validate_partition(global_parameters):
    partition_path = global_parameters['partition_data']
    if not file_util.file_exists(partition_path):
        raise ValueError('Partition does not exist')
    with h5py.File(partition_path, 'r') as partition_h5:
        if 'train' not in partition_h5.keys():
            raise ValueError('Partition does not contain train data')
        if 'test' not in partition_h5.keys():
            raise ValueError('Partition does not contain test data')


def validate_preprocessed(global_parameters):
    preprocessed_path = global_parameters['preprocessed_data']
    if not file_util.file_exists(preprocessed_path):
        raise ValueError('Preprocessed does not exist')
    with h5py.File(preprocessed_path, 'r') as preprocessed_h5:
        if 'preprocessed' not in preprocessed_h5.keys():
            raise ValueError('Preprocessed does not contain preprocessed data')


def validate_network(global_parameters):
    network_path = file_structure.get_network_file(global_parameters)
    if not file_util.file_exists(network_path):
        raise ValueError('Network does not exist')
    if models.load_model(network_path) is None:
        raise ValueError('Could not load network')


def validate_preprocessed_images(global_parameters):
    if not file_util.file_exists(global_parameters['preprocessed_data']):
        raise ValueError('Preprocessed data does not exist')
    if len(file_util.list_files(global_parameters['preprocessed_data'])) < global_parameters['n']:
        raise ValueError('Less files in preprocessed data than expected')


def validate_prediction(global_parameters):
    prediction_path = file_structure.get_prediction_file(global_parameters)
    if not file_util.file_exists(prediction_path):
        raise ValueError('Prediction does not exist')
    with h5py.File(prediction_path, 'r') as prediction_h5:
        if 'train' not in prediction_h5.keys():
            raise ValueError('Prediction does not contain train data')
        if 'test' not in prediction_h5.keys():
            raise ValueError('Prediction does not contain test data')
