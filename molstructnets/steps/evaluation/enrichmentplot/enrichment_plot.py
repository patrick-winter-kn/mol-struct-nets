from util import data_validation


class EnrichmentPlot:

    @staticmethod
    def get_id():
        return 'enrichment_plot'

    @staticmethod
    def get_name():
        return 'Enrichment Plot'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id':'enrichment_factors', 'name':'Enrichment Factors (in %, default: 5,10)', 'type':str, 'default':'5,10'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_prediction(global_parameters)

    @staticmethod
    def execute():
        # TODO
        pass
