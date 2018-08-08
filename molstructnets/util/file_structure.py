import re

from util import file_util, constants


class DataSet:
    smiles = 'smiles'


class Target:
    classes = 'classes'


class Partitions:
    train = 'train'
    test = 'test'


class Preprocessed:
    atom_locations = 'atom_locations'
    index = 'index'
    preprocessed = 'preprocessed'
    preprocessed_normalization_stats = 'preprocessed_normalization_stats'


class PreprocessedTraining:
    preprocessed_training = 'preprocessed_training'
    preprocessed_training_references = 'preprocessed_training_references'


class Predictions:
    prediction = 'prediction'


class Cam:
    substructure_atoms = 'substructure_atoms'
    cam_active = 'cam_active'
    cam_inactive = 'cam_inactive'
    cam_active_indices = 'cam_active_indices'
    cam_inactive_indices = 'cam_inactive_indices'


class PreprocessedTensor2DJit:
    dimensions = 'dimensions'
    min_x = 'min_x'
    max_x = 'max_x'
    min_y = 'min_y'
    max_y = 'max_y'
    square = 'square'
    scale = 'scale'
    with_bonds = 'with_bonds'
    symbols = 'symbols'
    chemical_properties = 'chemical_properties'
    gauss_sigma = 'gauss_sigma'
    normalization_type = 'normalization_type'
    normalization_min = 'normalization_min'
    normalization_max = 'normalization_max'
    normalization_mean = 'normalization_mean'
    normalization_std = 'normalization_std'


class CamSubstructures:
    active_substructures = 'active_substructures'
    active_substructures_occurrences = 'active_substructures_occurrences'
    active_substructures_value = 'active_substructures_value'
    active_substructures_number_heavy_atoms = 'active_substructures_number_heavy_atoms'
    active_substructures_score = 'active_substructures_score'
    inactive_substructures = 'inactive_substructures'
    inactive_substructures_occurrences = 'inactive_substructures_occurrences'
    inactive_substructures_value = 'inactive_substructures_value'
    inactive_substructures_number_heavy_atoms = 'inactive_substructures_number_heavy_atoms'
    inactive_substructures_score = 'inactive_substructures_score'


def get_root_from_experiment_file(experiment_file_path):
    return file_util.get_parent(file_util.get_parent(experiment_file_path))


def get_experiment_folder(global_parameters):
    return file_util.resolve_subpath(global_parameters[constants.GlobalParameters.root],
                                     'experiments')


def get_data_set_folder(global_parameters):
    return file_util.resolve_subpath(global_parameters[constants.GlobalParameters.root],
                                     'data_sets')


def get_data_set_file(global_parameters):
    return file_util.resolve_subpath(global_parameters[constants.GlobalParameters.root],
                                     'data_sets',
                                     global_parameters[constants.GlobalParameters.data_set] + '.h5')


def get_target_folder(global_parameters):
    return file_util.resolve_subpath(global_parameters[constants.GlobalParameters.root],
                                     'data_sets',
                                     global_parameters[constants.GlobalParameters.data_set],
                                     'targets')


def get_target_file(global_parameters):
    return file_util.resolve_subpath(global_parameters[constants.GlobalParameters.root],
                                     'data_sets',
                                     global_parameters[constants.GlobalParameters.data_set],
                                     'targets',
                                     global_parameters[constants.GlobalParameters.target] + '.h5')


def get_partition_folder(global_parameters):
    return file_util.resolve_subpath(global_parameters[constants.GlobalParameters.root],
                                     'data_sets',
                                     global_parameters[constants.GlobalParameters.data_set],
                                     'targets',
                                     global_parameters[constants.GlobalParameters.target],
                                     'partitions')


def get_partition_file(global_parameters):
    return file_util.resolve_subpath(global_parameters[constants.GlobalParameters.root],
                                     'data_sets',
                                     global_parameters[constants.GlobalParameters.data_set],
                                     'targets',
                                     global_parameters[constants.GlobalParameters.target],
                                     'partitions',
                                     global_parameters[constants.GlobalParameters.partition_data] + '.h5')


def get_preprocessed_folder(global_parameters):
    return file_util.resolve_subpath(global_parameters[constants.GlobalParameters.root],
                                     'data_sets',
                                     global_parameters[constants.GlobalParameters.data_set],
                                     'preprocessed')


def get_preprocessed_training_folder(global_parameters):
    partition = file_util.get_filename(global_parameters[constants.GlobalParameters.partition_data], False)
    preprocessed = global_parameters[constants.GlobalParameters.preprocessed_data]
    return file_util.resolve_subpath(global_parameters[constants.GlobalParameters.root],
                                     'data_sets',
                                     global_parameters[constants.GlobalParameters.data_set],
                                     'targets',
                                     global_parameters[constants.GlobalParameters.target],
                                     'partitions',
                                     partition,
                                     'preprocessed',
                                     preprocessed[preprocessed.rfind('/') + 1:preprocessed.rfind('.')])


def get_network_file(global_parameters):
    return file_util.resolve_subpath(get_result_folder(global_parameters),
                                     feature_prefix(global_parameters) + 'network.h5')


def get_random_forest_file(global_parameters):
    return file_util.resolve_subpath(get_result_folder(global_parameters),
                                     feature_prefix(global_parameters) + 'randomforest.pkl.gz')


def get_prediction_file(global_parameters):
    return file_util.resolve_subpath(get_result_folder(global_parameters),
                                     feature_prefix(global_parameters) + 'predictions.h5')


def get_evaluation_folder(global_parameters):
    return file_util.resolve_subpath(get_result_folder(global_parameters),
                                     feature_prefix(global_parameters) + 'evaluation')


def get_evaluation_stats_file(global_parameters):
    return file_util.resolve_subpath(get_evaluation_folder(global_parameters),
                                     'stats.csv')


def get_interpretation_folder(global_parameters):
    return file_util.resolve_subpath(get_result_folder(global_parameters),
                                     feature_prefix(global_parameters) + 'interpretation')


def get_cam_file(global_parameters):
    return file_util.resolve_subpath(get_interpretation_folder(global_parameters), 'cam.h5')


def get_commit_hash_file(global_parameters):
    return file_util.resolve_subpath(get_result_folder(global_parameters),
                                     'commit_hash.txt')


def get_result_folder(global_parameters):
    return file_util.resolve_subpath(global_parameters[constants.GlobalParameters.root],
                                     'experiments',
                                     global_parameters[constants.GlobalParameters.experiment],
                                     global_parameters[constants.GlobalParameters.data_set],
                                     global_parameters[constants.GlobalParameters.target],
                                     global_parameters[constants.GlobalParameters.partition_data],
                                     str(global_parameters[constants.GlobalParameters.seed]))


def feature_prefix(global_parameters):
    if constants.GlobalParameters.feature_id in global_parameters:
        return global_parameters[constants.GlobalParameters.feature_id] + '_'
    else:
        return ''



def find_file(folder_path, name):
    prefix = file_util.resolve_subpath(folder_path, name)
    if file_util.file_exists(prefix + '.h5'):
        return name
    files = file_util.list_files(folder_path)
    pattern = re.compile(re.escape(prefix) + '.*')
    for file in files.copy():
        if file_util.is_folder(file):
            files.remove(file)
        if not pattern.match(file):
            files.remove(file)
    if len(files) == 0:
        raise IOError('No file with prefix ' + name + ' found in ' + folder_path)
    elif len(files) > 1:
        raise IOError('Multiple files with prefix ' + name + ' found in ' + folder_path)
    return file_util.get_filename(files[0], False)


def find_data_set(global_parameters_, name):
    data_set_folder = get_data_set_folder(global_parameters_)
    return find_file(data_set_folder, name)


def find_target(global_parameters_, name):
    target_folder = get_target_folder(global_parameters_)
    return find_file(target_folder, name)


def find_partition(global_parameters_, name):
    partition_folder = get_partition_folder(global_parameters_)
    return find_file(partition_folder, name)
