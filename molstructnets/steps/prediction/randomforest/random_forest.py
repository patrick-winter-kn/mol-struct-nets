import h5py
from sklearn.externals import joblib

from steps.training.shared.randomforest import random_forest
from util import data_validation, file_structure, logger, file_util, constants, hdf5_util


class RandomForest:

    @staticmethod
    def get_id():
        return 'random_forest'

    @staticmethod
    def get_name():
        return 'Random Forest'

    @staticmethod
    def get_parameters():
        return list()

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed(global_parameters)
        random_forest_path = file_util.resolve_subpath(file_structure.get_result_folder(global_parameters),
                                                       'randomforest.pkl.gz')
        if not file_util.file_exists(random_forest_path):
            raise ValueError('File ' + random_forest_path + ' does not exist.')

    @staticmethod
    def execute(global_parameters, local_parameters):
        prediction_path = file_structure.get_prediction_file(global_parameters)
        if file_util.file_exists(prediction_path):
            logger.log('Skipping step: ' + prediction_path + ' already exists')
        else:
            preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data], 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed][:]
            preprocessed_h5.close()
            temp_prediction_path = file_util.get_temporary_file_path('random_forest_prediction')
            model_path = file_util.resolve_subpath(file_structure.get_result_folder(global_parameters),
                                                   'randomforest.pkl.gz')
            model = joblib.load(model_path)
            predictions = random_forest.predict(preprocessed, model)
            prediction_h5 = h5py.File(temp_prediction_path, 'w')
            hdf5_util.create_dataset_from_data(prediction_h5, file_structure.Predictions.prediction, predictions)
            prediction_h5.close()
            file_util.move_file(temp_prediction_path, prediction_path)
