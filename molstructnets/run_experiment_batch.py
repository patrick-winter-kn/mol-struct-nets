import argparse
import subprocess
import sys

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
experiments, seeds = experiment_batch.load_entries_from_csv(args.batch_csv)
nr_seeds = 1
if seeds is not None:
    nr_seeds = len(seeds)
results = execution_results.ExecutionResults(result_path, len(experiments))
run_experiment = [sys.executable, sys.argv[0][:sys.argv[0].rfind('/') + 1] + 'run_experiment.py']
logger.log('\n')
i = 0
while i < len(experiments):
    if results.get_status(i) != execution_results.Status.success:
        successes = 0
        for j in range(nr_seeds):
            params = run_experiment + experiments[i].get_execution_arguments()
            if seeds is not None:
                params += ['--seed', str(seeds[j])]
            retry_text = ''
            for k in range(args.retries + 1):
                logger.divider('•', nr_lines=2)
                logger.log(retry_text + ' '.join(params))
                logger.divider('•', nr_lines=2)
                logger.log('\n')
                result = subprocess.call(params)
                logger.log('\n')
                if result != 0:
                    results.set_status(i, execution_results.Status.failed)
                    retry_text = 'Running for the ' + str(k + 2) + '. time:\n'
                else:
                    successes += 1
                    break
            if successes == nr_seeds:
                results.set_status(i, execution_results.Status.success)
            results.save()
    experiments, seeds = experiment_batch.load_entries_from_csv(args.batch_csv)
    nr_seeds = 1
    if seeds is not None:
        nr_seeds = len(seeds)
    results.update_number_experiments(len(experiments))
    i += 1
