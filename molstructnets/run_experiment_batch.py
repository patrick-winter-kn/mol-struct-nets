import sys
import argparse
import subprocess
from experimentbatch import experiment_batch, execution_results
from util import file_util, logger


def get_arguments():
    parser = argparse.ArgumentParser(description='Runs a batch of experiments')
    parser.add_argument('batch_csv', type=str, help='CSV file containing the experiment parameters in the form: '
                                                    'experiment, data set, target, partition. Unused parameters can be '
                                                    'left empty.')
    parser.add_argument('--retries', type=int, default=0, help='Number of retries if an experiment fails')
    return parser.parse_args()


args = get_arguments()
result_path = file_util.resolve_path(args.batch_csv[:args.batch_csv.rfind('.')] + '_execution_results.csv')
experiments = experiment_batch.load_entries_from_csv(args.batch_csv)
results = execution_results.ExecutionResults(result_path, len(experiments))
run_experiment = sys.argv[0][:sys.argv[0].rfind('/')]
for i in range(results.size()):
    params = [run_experiment] + experiments[i].get_execution_arguments()
    logger.log('=' * 100 + '\n' + ' '.join(params) + '=' * 100 + '\n' * 3)
    if results.get_status(i) != execution_results.Status.success:
        for j in range(args.retries + 1):
            result = subprocess.call(params)
            if result != 0:
                results.set_status(i, execution_results.Status.failed)
            else:
                results.set_status(i, execution_results.Status.success)
                break
