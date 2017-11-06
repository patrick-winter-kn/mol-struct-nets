from util import initialization
import os
from os import path
import sys
import argparse
from experiments import experiment
from util import file_structure, constants, file_util
from steps import steps_repository
from steps.datageneration import data_generation_repository
from steps.targetgeneration import target_generation_repository
import threading
import time


def get_arguments():
    parser = argparse.ArgumentParser(description='Shows the tensorboard of an existing experiment')
    parser.add_argument('experiment', type=str, help='Path to the experiment file')
    parser.add_argument('--data_set', type=str, default=None, help='Data set name')
    parser.add_argument('--target', type=str, default=None, help='Target name')
    parser.add_argument('--partition', type=str, default=None, help='Partition name')
    return parser.parse_args()


def open_tensorboard():
    time.sleep(2)
    os.system('xdg-open http://localhost:6006')


args = get_arguments()
executable_path = sys.executable[:sys.executable.rfind(path.sep) + 1]
experiment_ = experiment.Experiment(args.experiment)
global_parameters = dict()
global_parameters[constants.GlobalParameters.seed] = initialization.seed
global_parameters[constants.GlobalParameters.root] = file_structure.get_root_from_experiment_file(args.experiment)
global_parameters[constants.GlobalParameters.experiment] = experiment_.get_name()
if args.data_set is not None:
    global_parameters[constants.GlobalParameters.data_set] = file_structure.find_data_set(global_parameters, args.data_set)
if args.target is not None:
    global_parameters[constants.GlobalParameters.target] = file_structure.find_target(global_parameters, args.target)
if args.partition is not None:
    global_parameters[constants.GlobalParameters.partition_data] = file_structure.find_partition(global_parameters, args.partition)
nr_steps = experiment_.number_steps()
for i in range(nr_steps):
    step_config = experiment_.get_step(i)
    type_id = step_config['type']
    step = steps_repository.instance.get_step_implementation(step_config['type'], step_config['id'])
    parameters = {}
    implementation_parameters = step.get_parameters()
    for parameter in implementation_parameters:
        if 'default' in parameter:
            parameters[parameter['id']] = parameter['default']
    if 'parameters' in step_config:
        for parameter in step_config['parameters']:
            parameters[parameter] = step_config['parameters'][parameter]
    if type_id == data_generation_repository.instance.get_id():
        global_parameters[constants.GlobalParameters.data_set] = file_util.get_filename(step.get_result_file(global_parameters, parameters), False)
    if type_id == target_generation_repository.instance.get_id():
        global_parameters[constants.GlobalParameters.target] = file_util.get_filename(step.get_result_file(global_parameters, parameters), False)
tensorboard_path = file_structure.get_network_file(global_parameters)
tensorboard_path = tensorboard_path[:tensorboard_path.rfind('.')] + '-tensorboard'
threading.Thread(target=open_tensorboard).start()
os.system(executable_path + 'tensorboard --logdir=' + tensorboard_path)
