from util import data_validation


class Substructure:

    @staticmethod
    def get_id():
        return 'substructure'

    @staticmethod
    def get_name():
        return 'Substructure'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id':'substructure', 'name':'Substructure', 'type':str})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def execute():
        # TODO
        pass
