from util import file_util

class BatchEntry:

    def __init__(self, csv_line, experiment_location=None):
        values = csv_line.split(',')
        self.experiment_location = experiment_location
        self.experiment = BatchEntry.get_value(values, 0)
        if self.experiment is None:
            raise ValueError('Experiment is not defined')
        self.data_set = BatchEntry.get_value(values, 1)
        self.target = BatchEntry.get_value(values, 2)
        self.partition = BatchEntry.get_value(values, 3)

    def get_execution_arguments(self):
        arguments = list()
        if self.experiment_location is not None:
            arguments.append(file_util.resolve_subpath(self.experiment_location, self.experiment))
        else:
            arguments.append(file_util.resolve_path(self.experiment))
        arguments.append(self.experiment)
        if self.data_set is not None:
            arguments.append('--data_set ' + self.data_set)
        if self.target is not None:
            arguments.append('--target ' + self.target)
        if self.partition is not None:
            arguments.append('--partition' + self.partition)
        return arguments

    @staticmethod
    def get_value(values, index):
        if len(values) < index + 1:
            return None
        value = values[index].strip()
        if value == '':
            return None
        return value
