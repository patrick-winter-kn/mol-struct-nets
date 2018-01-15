from util import file_util
from experimentbatch import batch_entry


def load_entries_from_csv(csv_path):
    with open(file_util.resolve_path(csv_path), 'r') as file:
        csv = file.read()
    experiment_location = None
    if csv.startswith('#'):
        experiment_location = csv[1:csv.find('\n')]
        csv = csv[csv.find('\n') + 1:]
        if experiment_location == 'same':
            experiment_location = csv_path[:csv_path.rfind('/')]
    entries = list()
    for line in csv.splitlines():
        entries.append(batch_entry.BatchEntry(line, experiment_location))
    return entries
