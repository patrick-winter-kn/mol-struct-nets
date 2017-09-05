from util import initialization
from keras import backend
import gc
import time
import argparse
from experiments import experiment
from util import file_structure, logger, file_util, constants
from steps import steps_repository
import h5py
import re


def get_arguments():
    parser = argparse.ArgumentParser(description='Runs an existing experiment')
    parser.add_argument('experiment', type=str, help='Path to the experiment file')
    parser.add_argument('--data_set', type=str, default=None, help='Data set name')
    parser.add_argument('--target', type=str, default=None, help='Target name')
    parser.add_argument('--step', type=int, default=None, help='Run the experiment up to the given step')
    return parser.parse_args()


def get_n(global_parameters_):
    n_ = None
    data_set_path = file_structure.get_data_set_file(global_parameters_)
    if file_util.file_exists(data_set_path):
        data_h5 = h5py.File(data_set_path)
        if file_structure.DataSet.smiles in data_h5.keys():
            n_ = len(data_h5[file_structure.DataSet.smiles])
        data_h5.close()
    return n_


def find_file(global_parameters_, folder_path, name):
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
    data_set_folder = file_structure.get_data_set_folder(global_parameters_)
    return find_file(global_parameters_, data_set_folder, name)


def find_target(global_parameters_, name):
    target_folder = file_structure.get_target_folder(global_parameters_)
    return find_file(global_parameters_, target_folder, name)


args = get_arguments()
experiment_ = experiment.Experiment(args.experiment)
global_parameters = dict()
global_parameters[constants.GlobalParameters.seed] = initialization.seed
global_parameters[constants.GlobalParameters.root] = file_structure.get_root_from_experiment_file(args.experiment)
global_parameters[constants.GlobalParameters.experiment] = experiment_.get_name()
if args.data_set is not None:
    global_parameters[constants.GlobalParameters.data_set] = find_data_set(global_parameters, args.data_set)
if args.target is not None:
    global_parameters[constants.GlobalParameters.target] = find_target(global_parameters, args.target)
n = get_n(global_parameters)
if n is not None:
    global_parameters[constants.GlobalParameters.n] = n
nr_steps = experiment_.number_steps()
if args.step is not None:
    nr_steps = args.step + 1
for i in range(nr_steps):
    step_config = experiment_.get_step(i)
    type_name = steps_repository.instance.get_step_name(step_config['type'])
    step = steps_repository.instance.get_step_implementation(step_config['type'], step_config['id'])
    logger.log('=' * 100)
    logger.log('Starting step: ' + type_name + ': ' + step.get_name())
    parameters = {}
    implementation_parameters = step.get_parameters()
    for parameter in implementation_parameters:
        if 'default' in parameter:
            parameters[parameter['id']] = parameter['default']
    if 'parameters' in step_config:
        for parameter in step_config['parameters']:
            parameters[parameter] = step_config['parameters'][parameter]
    step.check_prerequisites(global_parameters, parameters)
    step.execute(global_parameters, parameters)
    logger.log('Finished step: ' + type_name + ': ' + step.get_name())
    backend.clear_session()
logger.log('=' * 100)
logger.log('Finished execution of experiment successfully')
gc.collect()
time.sleep(1)
