from util import data_validation


class SmilesAttention:

    @staticmethod
    def get_id():
        return 'smiles_attention'

    @staticmethod
    def get_name():
        return 'SMILES Attention'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'top_n', 'name': 'Top n (default: all)', 'type': int, 'default': None})
        parameters.append({'id': 'actives', 'name': 'Active class (otherwise inactive, default: True)', 'type': bool,
                           'default': True})
        parameters.append({'id': 'correct_predictions', 'name': 'Only correct predictions (default: False)',
                           'type': bool, 'default': False})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed(global_parameters)
        data_validation.validate_network(global_parameters)

    @staticmethod
    def execute(global_parameters, parameters):
        # TODO
        pass
