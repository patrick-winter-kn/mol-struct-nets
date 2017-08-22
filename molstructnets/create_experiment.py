import argparse
from experiments import experiment
from util import file_util, file_structure
from steps import steps_repository


def get_arguments():
    parser = argparse.ArgumentParser(description='Creates a new experiment')
    return parser.parse_args()


def list_selections(options):
    for i in range(len(options)):
        print(str(i) + ': ' + options[i])

# TODO handle input errors

args = get_arguments()
path = input('Root folder: ')
name = input('Experiment name: ')
new_experiment = experiment.Experiment(file_util.resolve_subpath(file_structure.get_experiment_folder({'root': path}),
                                                                 name+'.json'))
steps = ['Done'] + steps_repository.instance.get_step_names()
selected_step = None
while selected_step != 0:
    print()
    list_selections(steps)
    selected_step = input('Step: ')
    selected_step = int(selected_step)
    if selected_step > 0:
        step = {}
        type_ = steps_repository.instance.get_steps()[selected_step-1].get_id()
        step['type'] = type_
        implementations = steps_repository.instance.get_step_implementation_names(type_)
        print()
        list_selections(implementations)
        selected_implementation = input('Implementation: ')
        selected_implementation = int(selected_implementation)
        implementation_id = steps_repository.instance.get_step_implementations(type_)[selected_implementation].get_id()
        step['id'] = implementation_id
        implementation = steps_repository.instance.get_step_implementation(type_, implementation_id)
        parameters = implementation.get_parameters()
        for parameter in parameters:
            value = input(parameter['name'] + ': ')
            if value is not '':
                value = parameter['type'](value)
                if 'parameters' not in step:
                    step['parameters'] = {}
                step['parameters'][parameter['id']] = value
            # TODO parameter explanation
            # TODO check if parameter can be optional
            # TODO special case for bool
        new_experiment.add_step(step)
print(str(new_experiment))
new_experiment.save()
