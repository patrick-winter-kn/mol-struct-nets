from util import initialization
from keras import backend
import gc
import time
import argparse
from experiments import experiment
from util import file_structure
from steps import steps_repository


def get_arguments():
    parser = argparse.ArgumentParser(description='Runs an existing experiment')
    parser.add_argument('experiment', type=str, help='Path to the experiment file')
    parser.add_argument('data_set', type=str, help='Data set name')
    parser.add_argument('target', type=str, help='Target name')
    parser.add_argument('--step', type=int, default=None, help='Run the experiment up to the given step')
    return parser.parse_args()


args = get_arguments()
experiment_ = experiment.Experiment(args.experiment)
global_parameters = {}
global_parameters['seed'] = initialization.seed
global_parameters['root'] = file_structure.get_root_from_experiment_file(args.experiment)
global_parameters['experiment'] = experiment_.get_name()
global_parameters['data_set'] = args.data_set
global_parameters['target'] = args.target
nr_steps = experiment_.number_steps()
if args.step is not None:
    nr_steps = args.step + 1
for i in range(nr_steps):
    step_config = experiment_.get_step(i)
    type_name = steps_repository.instance.get_step_name(step_config['type'])
    step = steps_repository.instance.get_step_implementation(step_config['type'], step_config['id'])
    print('=' * 100)
    print('Starting step: ' + type_name + ': ' + step.get_name())
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
    print('Finished step: ' + type_name + ': ' + step.get_name())
    backend.clear_session()
print('=' * 100)
print('Finished execution of experiment successfully')
gc.collect()
time.sleep(1)
