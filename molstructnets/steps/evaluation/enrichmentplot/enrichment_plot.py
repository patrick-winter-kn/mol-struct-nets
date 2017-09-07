import h5py

from util import data_validation, misc, file_util, file_structure, logger, reference_data_set, constants, enrichment


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
        parameters.append({'id': 'method_name', 'name': 'Method name', 'type': str,
                           'description': 'Name of the evaluated method that will be shown in the plot.'})
        parameters.append({'id': 'enrichment_factors', 'name': 'Enrichment Factors (in %, default: 5,10)', 'type': str,
                           'default': '5,10', 'description': 'List of enrichment factors in percent.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_prediction(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(local_parameters, ['enrichment_factors'])
        file_name = 'enrichment_plot_' + misc.hash_parameters(hash_parameters) + '.svg'
        return file_util.resolve_subpath(file_structure.get_evaluation_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        enrichment_plot_path = EnrichmentPlot.get_result_file(global_parameters, local_parameters)
        if file_util.file_exists(enrichment_plot_path):
            logger.log('Skipping step: ' + enrichment_plot_path + ' already exists')
        else:
            enrichment_factors = []
            for enrichment_factor in local_parameters['enrichment_factors'].split(','):
                enrichment_factors.append(int(enrichment_factor))
            partition_h5 = h5py.File(global_parameters[constants.GlobalParameters.partition_data], 'r')
            test = partition_h5[file_structure.Partitions.test]
            target_h5 = h5py.File(file_structure.get_target_file(global_parameters), 'r')
            classes = target_h5[file_structure.Target.classes]
            ground_truth = reference_data_set.ReferenceDataSet(test, classes)
            prediction_h5 = h5py.File(file_structure.get_prediction_file(global_parameters), 'r')
            predictions = reference_data_set.ReferenceDataSet(test,
                                                              prediction_h5[file_structure.Predictions.prediction])
            enrichment.plot([predictions], [local_parameters['method_name']], ground_truth, enrichment_factors,
                            enrichment_plot_path)
            partition_h5.close()
            target_h5.close()
            prediction_h5.close()
