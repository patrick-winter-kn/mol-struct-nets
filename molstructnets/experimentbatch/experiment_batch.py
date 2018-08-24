from experimentbatch import batch_entry
from util import file_util


def load_entries_from_csv(csv_path):
    with open(file_util.resolve_path(csv_path), 'r') as file:
        csv = file.read().strip()
    experiment_location = None
    entries = list()
    seeds = None
    for line in csv.splitlines():
        if line.startswith('#'):
            if line.startswith('# location'):
                experiment_location = csv[11:csv.find('\n')]
                csv = csv[csv.find('\n') + 1:]
                if experiment_location == 'same':
                    experiment_location = csv_path[:csv_path.rfind('/')]
            elif line.startswith('# seeds'):
                seeds = csv[8:csv.find('\n')]
                if '-' in seeds:
                    seeds = seeds.split('-')
                    seeds = list(range(int(seeds[0]), int(seeds[1]) + 1))
                elif ',' in seeds:
                    seeds = seeds.split(',')
                    seeds = list(map(int, seeds))
                else:
                    seeds = [seeds]
        else:
            entries.append(batch_entry.BatchEntry(line, experiment_location))
    return entries, seeds
