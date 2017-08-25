from util import data_validation, file_structure, misc, file_util


class StratifiedSampling:

    @staticmethod
    def get_id():
        return 'stratified_sampling'

    @staticmethod
    def get_name():
        return 'Stratified Sampling'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'train_percentage', 'name': 'Size of training partition (in %)', 'type': int})
        parameters.append({'id': 'oversample', 'name': 'Oversample training partition', 'type': bool})
        parameters.append({'id': 'shuffle', 'name': 'Shuffle training partition', 'type': bool})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_target(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, parameters):
        hash_parameters = misc.copy_dict_from_keys(global_parameters, ['seed'])
        hash_parameters.update(misc.copy_dict_from_keys(parameters, ['train_percentage', 'oversample', 'shuffle']))
        file_name = 'stratified_sampling_' + misc.hash_parameters(hash_parameters) + '.h5'
        return file_util.resolve_subpath(file_structure.get_partition_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, parameters):
        # TODO
        pass
