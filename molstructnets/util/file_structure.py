from util import file_util


def get_experiment_folder(global_parameters):
    return file_util.resolve_subpath(global_parameters['root'], 'experiments')


def get_data_set_file(global_parameters):
    return file_util.resolve_subpath(global_parameters['root'], ['data_sets', global_parameters['data_set'] + '.h5'])


def get_target_file(global_parameters):
    return file_util.resolve_subpath(global_parameters['root'], ['data_sets', global_parameters['data_set'], 'targets', global_parameters['target'] + '.h5'])
