import argparse
from experiments import experiment


def get_arguments():
    parser = argparse.ArgumentParser(description='Shows the steps of an existing experiment')
    parser.add_argument('experiment', type=str, help='Path to the experiment file')
    return parser.parse_args()


args = get_arguments()
experiment = experiment.Experiment(args.experiment)
print(experiment)
