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
        parameters.append({'id': 'nr_trees', 'name': 'Number of Trees', 'type': int, 'min': 1, 'default': 1000,
                           'description': 'The number of trees in the random forest. Default: 1000'})
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
            target_h5 = h5py.File(file_structure.get_target_file(global_parameters), 'r')
            classes = target_h5[file_structure.Target.classes]
            preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data], 'r')
            partition_h5 = h5py.File(file_structure.get_partition_file(global_parameters), 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            preprocessed_training_h5 = None
            if constants.GlobalParameters.preprocessed_training_data in global_parameters:
                preprocessed_training_h5 =\
                    h5py.File(global_parameters[constants.GlobalParameters.preprocessed_training_data], 'r')
                train = preprocessed_training_h5[file_structure.PreprocessedTraining.preprocessed_training_references]
                input_ = preprocessed_training_h5[file_structure.PreprocessedTraining.preprocessed_training]
            else:
                train = partition_h5[file_structure.Partitions.train]
                input_ = reference_data_set.ReferenceDataSet(train, preprocessed)
            output = reference_data_set.ReferenceDataSet(train, classes)
            input_ = misc.copy_into_memory(input_, as_bool=True)
            output = misc.copy_into_memory(output, as_bool=True)
            random_forest.train(input_, output, model_path, local_parameters['nr_trees'])
            target_h5.close()
            preprocessed_h5.close()
            if preprocessed_training_h5 is not None:
                preprocessed_training_h5.close()
            if 'partition_h5' in locals():
                partition_h5.close()
