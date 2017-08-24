from util import file_util


def get_experiment_folder(global_parameters):
    return file_util.resolve_subpath(global_parameters['root'], 'experiments')


def get_data_set_file(global_parameters):
    return file_util.resolve_subpath(global_parameters['root'], ['data_sets', global_parameters['data_set'] + '.h5'])


def get_target_file(global_parameters):
    return file_util.resolve_subpath(global_parameters['root'], ['data_sets', global_parameters['data_set'], 'targets', global_parameters['target'] + '.h5'])


def get_partition_folder(global_parameters):
    return file_util.resolve_subpath(global_parameters['root'], ['data_sets', global_parameters['data_set'], 'targets', global_parameters['target'] + 'partitions'])


def get_preprocessed_folder(global_parameters):
    return file_util.resolve_subpath(global_parameters['root'], ['data_sets', global_parameters['data_set'], 'preprocessed'])


def get_network_file(global_parameters):
    return file_util.resolve_subpath(global_parameters['root'], ['experiments', global_parameters['experiment'], global_parameters['data_set'], global_parameters['target'], 'network.h5'])


def get_prediction_file(global_parameters):
    return file_util.resolve_subpath(global_parameters['root'], ['experiments', global_parameters['experiment'], global_parameters['data_set'], global_parameters['target'], 'predictions.h5'])


def get_root_from_experiment_file(experiment_file_path):
    return file_util.get_parent(file_util.get_parent(experiment_file_path))
