import json
import copy
from util import file_util
from steps import steps_repository


class Experiment:

    def __init__(self, file_path):
        self._file_path = file_util.resolve_path(file_path)
        if file_util.file_exists(self._file_path):
            self._dict = json.load(open(self._file_path))
        else:
            self._dict = {'steps': []}

    def save(self):
        file_util.make_folders(self._file_path)
        json.dump(self._dict, open(self._file_path, 'w+'), sort_keys=True, indent=4, separators=(',', ': '))

    def number_steps(self):
        return len(self._dict['steps'])

    def get_step(self, n):
        return copy.deepcopy(self._dict['steps'][n])

    def get_name(self):
        return file_util.get_filename(self._file_path, with_extension=False)

    def add_step(self, step):
        self._dict['steps'].append(step)

    def set_random_seed(self, seed):
        if seed is not None:
            self._dict['seed'] = seed
        elif 'seed' in self._dict:
            del self._dict['seed']

    def get_seed(self):
        if 'seed' in self._dict:
            return self._dict['seed']
        else:
            return None

    def __str__(self):
        string = ''
        string += self._file_path + '\n'
        string += '===== ' + self.get_name() + ' =====\n'
        if len(self._dict['steps']) > 0:
            for step in self._dict['steps']:
                type_name = steps_repository.instance.get_step_name(step['type'])
                implementation = steps_repository.instance.get_step_implementation(step['type'], step['id'])
                id_name = implementation.get_name()
                string += type_name + ': ' + id_name + '\n'
                if 'parameters' in step:
                    for parameter in implementation.get_parameters():
                        if parameter['id'] in step['parameters']:
                            string += '  ' + parameter['name'] + ': ' + str(step['parameters'][parameter['id']]) + '\n'
        string = string[:-1]
        return string
