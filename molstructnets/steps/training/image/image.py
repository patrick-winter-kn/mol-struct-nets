from util import data_validation


class Image:

    @staticmethod
    def get_id():
        return 'image'

    @staticmethod
    def get_name():
        return 'Image'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'epochs', 'name': 'Epochs', 'type': int})
        parameters.append({'id': 'batch_size', 'name': 'Batch size', 'type': int, 'default': 1})
        parameters.append({'id': 'validation', 'name': 'Validation', 'type': bool, 'default': False})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed_images(global_parameters)

    @staticmethod
    def execute(global_parameters, parameters):
        # TODO
        pass
