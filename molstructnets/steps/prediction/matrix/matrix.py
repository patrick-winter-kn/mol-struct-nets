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
        parameters.append({'id': 'batch_size', 'name': 'Batch size', 'type': int, 'default': 50})
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
