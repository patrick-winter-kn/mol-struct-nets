import argparse
from experiments import experiment
from util import file_util, file_structure, input_util
from steps import steps_repository


def get_arguments():
    parser = argparse.ArgumentParser(description='Creates a new experiment')
    return parser.parse_args()


def list_selections(options):
    for i in range(len(options)):
        print(str(i) + ': ' + options[i])


args = get_arguments()
path = input_util.read_string('*Root folder: ')
name = input_util.read_string('*Experiment name: ', regex='[a-zA-Z0-9_]+')
new_experiment = experiment.Experiment(file_util.resolve_subpath(file_structure.get_experiment_folder({'root': path}),
                                                                 name+'.json'))
seed = input_util.read_int('Random seed: ', True)
if seed is not None:
    new_experiment.set_random_seed(seed)
steps = steps_repository.instance.get_step_names()
selected_step = -1
while selected_step is not None:
    print()
    list_selections(steps)
    selected_step = input_util.read_int('Step: ', True, min_=0, max_=len(steps)-1)
    if selected_step is not None:
        step = {}
        type_ = steps_repository.instance.get_steps()[selected_step].get_id()
        step['type'] = type_
        implementations = steps_repository.instance.get_step_implementation_names(type_)
        print()
        list_selections(implementations)
        selected_implementation = input_util.read_int('*Implementation: ', min_=0, max_=len(implementations)-1)
        implementation_id = steps_repository.instance.get_step_implementations(type_)[selected_implementation].get_id()
        step['id'] = implementation_id
        implementation = steps_repository.instance.get_step_implementation(type_, implementation_id)
        parameters = implementation.get_parameters()
        for parameter in parameters:
            text = parameter['name'] + ': '
            optional = 'default' in parameter
            if not optional:
                text = '*' + text
            description = None
            if 'description' in parameter:
                description = parameter['description']
            if parameter['type'] is bool:
                value = input_util.read_bool(text, optional, description)
            elif parameter['type'] is int:
                value = input_util.read_int(text, optional, description)
            elif parameter['type'] is float:
                value = input_util.read_float(text, optional, description)
            else:
                value = input_util.read_string(text, optional, description)
            if value is not None:
                if 'parameters' not in step:
                    step['parameters'] = {}
                step['parameters'][parameter['id']] = value
        new_experiment.add_step(step)
print(str(new_experiment))
new_experiment.save()
