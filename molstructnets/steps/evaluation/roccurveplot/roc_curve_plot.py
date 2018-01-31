import h5py
import numpy
from steps.evaluation.shared import roc_curve
from util import data_validation, misc, file_util, file_structure, logger, reference_data_set, constants, csv_file


class RocCurvePlot:

    @staticmethod
    def get_id():
        return 'roc_curve_plot'

    @staticmethod
    def get_name():
        return 'ROC Curve Plot'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'method_name', 'name': 'Method Name', 'type': str, 'default': None,
                           'description': 'Name of the evaluated method that will be shown in the plot. Default:'
                                          ' Partition name'})
        parameters.append({'id': 'shuffle', 'name': 'Shuffle', 'type': bool,
                           'default': True, 'description': 'Shuffles the data before evaluation to counter sorted data'
                                                           ' sets, which can be a problem in cases where the'
                                                           ' probability is equal. Default: True'})
        parameters.append({'id': 'partition', 'name': 'Partition', 'type': str, 'default': 'test',
                           'options': ['train', 'test', 'both'],
                           'description': 'The enrichment plot will be generated for the specified partition. Default:'
                                          ' test'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_prediction(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(local_parameters, ['shuffle'])
        file_name = 'roc_curve_plot_' + local_parameters['partition'] + '-' + misc.hash_parameters(hash_parameters) +\
                    '.svgz'
        return file_util.resolve_subpath(file_structure.get_evaluation_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        roc_curve_plot_path = RocCurvePlot.get_result_file(global_parameters, local_parameters)
        if file_util.file_exists(roc_curve_plot_path):
            logger.log('Skipping step: ' + roc_curve_plot_path + ' already exists')
        else:
            method_name = local_parameters['method_name']
            if method_name is None:
                method_name = local_parameters['partition'].title()
            partition_h5 = h5py.File(file_structure.get_partition_file(global_parameters), 'r')
            target_h5 = h5py.File(file_structure.get_target_file(global_parameters), 'r')
            classes = target_h5[file_structure.Target.classes]
            prediction_h5 = h5py.File(file_structure.get_prediction_file(global_parameters), 'r')
            ground_truth = classes
            predictions = prediction_h5[file_structure.Predictions.prediction]
            partition = None
            if local_parameters['partition'] == 'train':
                partition = partition_h5[file_structure.Partitions.train]
                # Remove oversampling
                partition = numpy.unique(partition)
            elif local_parameters['partition'] == 'test' or local_parameters['partition'] != 'both':
                partition = partition_h5[file_structure.Partitions.test]
            if partition is not None:
                ground_truth = reference_data_set.ReferenceDataSet(partition, classes)
                predictions = reference_data_set.ReferenceDataSet(partition,
                                                                  prediction_h5[file_structure.Predictions.prediction])
            auc_list = roc_curve.plot([predictions], [method_name], ground_truth,
                                                 roc_curve_plot_path, local_parameters['shuffle'],
                                                 global_parameters[constants.GlobalParameters.seed])
            csv_path = file_structure.get_evaluation_stats_file(global_parameters)
            csv = csv_file.CsvFile(csv_path)
            row = dict()
            row['roc_curve_auc'] = auc_list[0]
            csv.add_row(method_name, row)
            csv.save()
            partition_h5.close()
            target_h5.close()
            prediction_h5.close()
