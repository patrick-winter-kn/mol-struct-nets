from util import data_validation, misc, file_structure, file_util


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
        parameters.append({'id': 'max_length', 'name': 'Maximum length (default: automatic)', 'type': int,
                           'default': None})
        parameters.append({'id': 'characters', 'name': 'Force characters (default: none)', 'type': str,
                           'default': None})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, parameters):
        hash_parameters = misc.copy_dict_from_keys(parameters, ['max_length', 'characters'])
        file_name = 'smiles_matrix_' + misc.hash_parameters(hash_parameters) + '.h5'
        return file_util.resolve_subpath(file_structure.get_preprocessed_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, parameters):
        # TODO
        raise NotImplementedError('This method has not yet been implemented')
