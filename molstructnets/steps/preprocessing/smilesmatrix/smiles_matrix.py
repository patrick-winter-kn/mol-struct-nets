from util import data_validation


class SmilesMatrix:

    @staticmethod
    def get_id():
        return 'smiles_matrix'

    @staticmethod
    def get_name():
        return 'SMILES Matrix'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'max_length', 'name': 'Maximum length (default: automatic)', 'type': int, 'default': None})
        parameters.append({'id': 'characters', 'name': 'Force characters (default: none)', 'type': str, 'default': None})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def execute():
        # TODO
        pass
