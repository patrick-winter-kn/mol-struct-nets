import argparse

from util import initialization


def get_arguments():
    parser = argparse.ArgumentParser(description='Runs an existing transfer learning experiment')
    parser.add_argument('experiment', type=str, help='Path to the experiment file')
    parser.add_argument('data_sets', type=str, help='Path to the data set JSON file')
    parser.add_argument('--seed', type=int, default=None, help='The random seed (will overwrite seed set in experiment'
                                                               ' file)')
    return parser.parse_args()


args = get_arguments()
initialization.initialize(args)

from keras import backend
import gc
import time
from experiments import experiment
from util import file_structure, logger, file_util, constants, misc, process_pool
from steps import steps_repository
from steps.preprocessing import preprocessing_repository
from steps.training import training_repository
from steps.trainingmultitarget import training_multitarget_repository
import datetime
import subprocess
import sys
import json


def add_last_commit_hash(script_path, global_parameters_):
    if misc.keys_in([constants.GlobalParameters.data_set, constants.GlobalParameters.target,
                     constants.GlobalParameters.partition_data], global_parameters_):
        commit_hash_path = file_structure.get_commit_hash_file(global_parameters_)
        if file_util.file_exists(file_util.get_parent(commit_hash_path)) \
                and not file_util.file_exists(commit_hash_path):
            git_repo_path = file_util.get_parent(script_path)
            hash_ = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=git_repo_path).decode('utf-8')[:-1]
            with open(commit_hash_path, 'w') as commit_hash_file:
                commit_hash_file.write(hash_)


def run_step(step, type_name, global_parameters, parameters):
        logger.divider()
        logger.header(type_name + ': ' + step.get_name())
        logger.header('Data set(s): ' + str(global_parameters[constants.GlobalParameters.data_set]))
        logger.log('')
        step.check_prerequisites(global_parameters, parameters)
        step.execute(global_parameters, parameters)
        process_pool.close_all_pools()
        backend.clear_session()
        gc.collect()


if not file_util.file_exists(args.experiment):
    logger.log('Experiment file ' + args.experiment + ' does not exist.', logger.LogLevel.ERROR)
    exit(1)
if not file_util.file_exists(args.data_sets):
    logger.log('Data sets file ' + args.args + ' does not exist.', logger.LogLevel.ERROR)
    exit(1)
start_time = datetime.datetime.now()
logger.divider()
logger.log('Starting experiment at ' + str(start_time))
experiment_ = experiment.Experiment(args.experiment)
global_parameters = dict()
global_parameters[constants.GlobalParameters.seed] = initialization.seed
global_parameters[constants.GlobalParameters.root] = file_structure.get_root_from_experiment_file(args.experiment)
global_parameters[constants.GlobalParameters.experiment] = experiment_.get_name()
transfer_data_sets = args.data_sets
transfer_data_sets = transfer_data_sets[transfer_data_sets.rfind('/') + 1 : transfer_data_sets.rfind('.')]
global_parameters[constants.GlobalParameters.transfer_data_sets] = transfer_data_sets
data_sets_object = json.loads(file_util.read_file(args.data_sets))
data_sets_train = data_sets_object['train']
for i in range(len(data_sets_train)):
    tmp_global_parameters = global_parameters.copy()
    data_sets_train[i][0] = file_structure.find_data_set(tmp_global_parameters, data_sets_train[i][0])
    tmp_global_parameters[constants.GlobalParameters.data_set] = data_sets_train[i][0]
    data_sets_train[i][1] = file_structure.find_target(tmp_global_parameters, data_sets_train[i][1])
data_sets_eval = data_sets_object['eval']
for i in range(len(data_sets_eval)):
    tmp_global_parameters = global_parameters.copy()
    data_sets_eval[i][0] = file_structure.find_data_set(tmp_global_parameters, data_sets_eval[i][0])
    tmp_global_parameters[constants.GlobalParameters.data_set] = data_sets_eval[i][0]
    data_sets_eval[i][1] = file_structure.find_target(tmp_global_parameters, data_sets_eval[i][1])
global_parameters_list = list()
for i in range(len(data_sets_train) + len(data_sets_eval)):
    if i < len(data_sets_train):
        tmp_data_set = data_sets_train[i]
    else:
        tmp_data_set = data_sets_eval[i - len(data_sets_train)]
    global_params = global_parameters.copy()
    global_params[constants.GlobalParameters.data_set] = tmp_data_set[0]
    global_params[constants.GlobalParameters.target] = tmp_data_set[1]
    global_parameters_list.append(global_params)
nr_steps = experiment_.number_steps()
for i in range(nr_steps):
    step_config = experiment_.get_step(i)
    type_id = step_config['type']
    type_name = steps_repository.instance.get_step_name(type_id)
    step = steps_repository.instance.get_step_implementation(step_config['type'], step_config['id'])
    parameters = {}
    implementation_parameters = step.get_parameters()
    for parameter in implementation_parameters:
        if 'default' in parameter:
            parameters[parameter['id']] = parameter['default']
    if 'parameters' in step_config:
        for parameter in step_config['parameters']:
            parameters[parameter] = step_config['parameters'][parameter]
    if type_id == preprocessing_repository.instance.get_id():
        data_sets = list()
        for j in range(len(data_sets_train)):
            data_sets.append(data_sets_train[j][0])
        for j in range(len(data_sets_eval)):
            data_sets.append(data_sets_eval[j][0])
        global_params = global_parameters.copy()
        global_params[constants.GlobalParameters.data_set] = data_sets
        run_step(step, type_name, global_params, parameters)
        for j in range(len(global_parameters_list)):
            global_parameters_list[j][constants.GlobalParameters.feature_id] = global_params[constants.GlobalParameters.feature_id]
            global_parameters_list[j][constants.GlobalParameters.preprocessed_data] = global_params[constants.GlobalParameters.preprocessed_data]
            global_parameters_list[j][constants.GlobalParameters.input_dimensions] = global_params[constants.GlobalParameters.input_dimensions]
    elif type_id == training_repository.instance.get_id():
        if len(data_sets_train) != 1:
            raise ValueError('Normal training does not support ' + str(len(data_sets_train)) + ' training data sets.')
        shared_network_path = file_structure.get_shared_network_file(global_parameters_list[0])
        if not file_util.file_exists(shared_network_path):
            file_util.copy_file(file_structure.get_network_file(global_parameters_list[0]), shared_network_path)
        global_parameters_list[0][constants.GlobalParameters.shared_network] = shared_network_path
        run_step(step, type_name, global_parameters_list[0], parameters)
        for j in range(1, len(global_parameters_list)):
            global_parameters_list[j][constants.GlobalParameters.shared_network] = global_parameters_list[0][constants.GlobalParameters.shared_network]
    elif type_id == training_multitarget_repository.instance.get_id():
        data_sets = list()
        for j in range(len(data_sets_train)):
            data_sets.append(data_sets_train[j])
        global_params = global_parameters.copy()
        global_params[constants.GlobalParameters.data_set] = data_sets
        run_step(step, type_name, global_params, parameters)
        for j in range(len(global_parameters_list)):
            global_parameters_list[j][constants.GlobalParameters.shared_network] = global_params[constants.GlobalParameters.shared_network]
    else:
        for j in range(len(data_sets_train) + len(data_sets_eval)):
            run_step(step, type_name, global_parameters_list[j], parameters)
logger.divider()
add_last_commit_hash(sys.argv[0], global_parameters)
end_time = datetime.datetime.now()
logger.log('Finished execution of experiment successfully at ' + str(end_time))
logger.log('Duration of experiment: ' + str(end_time - start_time))
logger.divider()
gc.collect()
time.sleep(1)
