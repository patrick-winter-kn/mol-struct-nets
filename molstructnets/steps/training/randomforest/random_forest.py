import h5py
from steps.training.shared.randomforest import random_forest
from util import data_validation, file_structure, file_util, logger, constants, reference_data_set, misc


class RandomForest:

    @staticmethod
    def get_id():
        return 'random_forest'

    @staticmethod
    def get_name():
        return 'Random Forest'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'nr_trees', 'name': 'Number of Trees', 'type': int, 'min': 1, 'default': 10000,
                           'description': 'The number of trees in the random forest. Default: 10000'})
        parameters.append({'id': 'min_samples_leaf', 'name': 'Minimum Samples per Leaf', 'type': int, 'min': 1,
                           'default': 10,
                           'description': 'The minimum number of samples contained in a leaf. Default: 10'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        return file_util.resolve_subpath(file_structure.get_result_folder(global_parameters), 'randomforest.pkl.gz')

    @staticmethod
    def execute(global_parameters, local_parameters):
        model_path = RandomForest.get_result_file(global_parameters, local_parameters)
        if file_util.file_exists(model_path):
            logger.log('Skipping step: ' + model_path + ' already exists')
        else:
            partition_h5 = h5py.File(file_structure.get_partition_file(global_parameters), 'r')
            train = partition_h5[file_structure.Partitions.train][:]
            partition_h5.close()
            target_h5 = h5py.File(file_structure.get_target_file(global_parameters), 'r')
            classes = target_h5[file_structure.Target.classes][:]
            target_h5.close()
            preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data], 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed][:]
            preprocessed_h5.close()
            classes = classes[train]
            preprocessed = preprocessed[train]
            random_forest.train(preprocessed, classes, model_path, local_parameters['nr_trees'],
                                local_parameters['min_samples_leaf'],
                                global_parameters[constants.GlobalParameters.seed])
