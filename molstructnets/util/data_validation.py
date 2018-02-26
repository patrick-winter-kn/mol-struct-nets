from util import file_util, file_structure, constants
import h5py
from keras import models


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


def validate_prediction(global_parameters):
    path = file_structure.get_prediction_file(global_parameters)
    validate_hdf5_file(path, file_structure.Predictions.prediction)


def validate_cam(global_parameters, *data_set_names):
    path = file_structure.get_cam_file(global_parameters)
    validate_hdf5_file(path, *data_set_names)


def validate_network(global_parameters):
    path = file_structure.get_network_file(global_parameters)
    if not file_util.file_exists(path):
        raise ValueError('File ' + path + ' does not exist')
    if models.load_model(path) is None:
        raise ValueError('Could not load network in ' + path)


def validate_preprocessed_images(global_parameters):
    path = global_parameters[constants.GlobalParameters.preprocessed_data]
    if not file_util.file_exists(path):
        raise ValueError('Folder ' + path + ' with preprocessed data does not exist')
    nr_files = len(file_util.list_files(path))
    n = global_parameters[constants.GlobalParameters.n]
    if nr_files < n:
        raise ValueError('Folder ' + path + ' with preprocessed data contains less files than expected ('
                         + str(nr_files) + '<' + str(n) + ')')


def validate_hdf5_file(path, *data_set_names):
    if not file_util.file_exists(path):
        raise ValueError('File ' + path + ' does not exist.')
    with h5py.File(path, 'r') as data_h5:
        for data_set_name in data_set_names:
            if data_set_name not in data_h5.keys():
                raise ValueError('File ' + path + ' does not contain data set \'' + data_set_name + '\'')
