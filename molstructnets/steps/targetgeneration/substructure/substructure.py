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
        parameters.append({'id':'substructures', 'name':'Substructures (separated by ;)', 'type':str})
        parameters.append({'id':'logic', 'name':'Logic expression (e.g. a&(b|c))', 'type':str, 'default':None})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def execute():
        # TODO
        pass
