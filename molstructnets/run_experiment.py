import argparse
from experiments import experiment
import random
import sys
from util import file_structure


def get_arguments():
    parser = argparse.ArgumentParser(description='Runs an existing experiment')
    parser.add_argument('experiment', type=str, help='Path to the experiment file')
    parser.add_argument('data_set', type=str, help='Data set name')
    parser.add_argument('target', type=str, help='Target name')
    parser.add_argument('--step', type=int, default=None, help='Run the experiment up to the given step')
    return parser.parse_args()


args = get_arguments()
experiment_ = experiment.Experiment(args.experiment)
global_variables = {}
seed = experiment_.get_seed()
if seed is None:
    seed = random.randint(0, sys.maxsize)
global_variables['seed'] = seed
global_variables['root'] = file_structure.get_root_from_experiment_file(args.experiment)
global_variables['experiment'] = experiment_.get_name()
global_variables['data_set'] = args.data_set
global_variables['target'] = args.target
# TODO