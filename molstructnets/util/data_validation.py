import h5py
from keras import models

from util import file_util, file_structure, constants


def validate_data_set(global_parameters):
    path = file_structure.get_data_set_file(global_parameters)
    validate_hdf5_file(path, file_structure.DataSet.smiles)


def validate_target(global_parameters):
    path = file_structure.get_target_file(global_parameters)
    validate_hdf5_file(path, file_structure.Target.classes)


def validate_partition(global_parameters):
    path = file_structure.get_partition_file(global_parameters)
    validate_hdf5_file(path, file_structure.Partitions.train, file_structure.Partitions.test)


def validate_preprocessed(global_parameters):
    path = global_parameters[constants.GlobalParameters.preprocessed_data]
    validate_hdf5_file(path, file_structure.Preprocessed.preprocessed)


def validate_preprocessed_specs(global_parameters):
    path = global_parameters[constants.GlobalParameters.preprocessed_data]
    validate_hdf5_file(path)


def validate_prediction(global_parameters):
    path = file_structure.get_prediction_file(global_parameters)
    validate_hdf5_file(path, file_structure.Predictions.prediction)


def validate_saliency_map(global_parameters, *data_set_names):
    path = file_structure.get_saliency_map_file(global_parameters)
    validate_hdf5_file(path, *data_set_names)


def validate_network(global_parameters):
    path = file_structure.get_network_file(global_parameters)
    if not file_util.file_exists(path):
        raise ValueError('File ' + path + ' does not exist')
    if models.load_model(path) is None:
        raise ValueError('Could not load network in ' + path)


def validate_hdf5_file(path, *data_set_names):
    if not file_util.file_exists(path):
        raise ValueError('File ' + path + ' does not exist.')
    with h5py.File(path, 'r') as data_h5:
        for data_set_name in data_set_names:
            if data_set_name not in data_h5.keys():
                raise ValueError('File ' + path + ' does not contain data set \'' + data_set_name + '\'')
