from util import data_validation, misc, file_structure, file_util


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
        parameters.append({'id': 'size', 'name': 'Size in pixels (n×n)', 'type': int})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, parameters):
        hash_parameters = misc.copy_dict_from_keys(parameters, ['size'])
        file_name = 'image_' + misc.hash_parameters(hash_parameters)
        return file_util.resolve_subpath(file_structure.get_preprocessed_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, parameters):
        # TODO
        pass
