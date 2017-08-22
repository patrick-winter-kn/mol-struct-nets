from util import data_validation


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
        parameters.append({'id':'train_percentage', 'name':'Size of training partition (in %)', 'type':int})
        parameters.append({'id':'oversample', 'name':'Oversample training partition', 'type':bool})
        parameters.append({'id':'shuffle', 'name':'Shuffle training partition', 'type':bool})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_target(global_parameters)

    @staticmethod
    def execute():
        # TODO
        pass
