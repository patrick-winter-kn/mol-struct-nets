from util import data_validation


class Matrix:

    @staticmethod
    def get_id():
        return 'matrix'

    @staticmethod
    def get_name():
        return 'Matrix'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id':'epochs', 'name':'Epochs', 'type':int})
        parameters.append({'id':'batch_size', 'name':'Batch size', 'type':int, 'default':50})
        parameters.append({'id':'validation', 'name':'Validation', 'type':bool, 'default':False})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed(global_parameters)

    @staticmethod
    def execute():
        # TODO
        pass
