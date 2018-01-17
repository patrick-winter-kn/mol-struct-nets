from util import data_validation, file_structure, progressbar, logger, file_util, constants, hdf5_util
import h5py
import math
from sklearn.externals import joblib
from steps.training.shared.randomforest import random_forest


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
        parameters.append({'id': 'batch_size', 'name': 'Batch Size', 'type': int, 'default': None, 'min': 1,
                           'description': 'Number of data points that will be processed together. A higher number leads'
                                          ' to faster processing but needs more memory. Default: all'})
        return parameters

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
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            temp_prediction_path = file_util.get_temporary_file_path('matrix_prediction')
            prediction_h5 = h5py.File(temp_prediction_path, 'w')
            predictions = hdf5_util.create_dataset(prediction_h5, file_structure.Predictions.prediction,
                                                   (len(preprocessed), 2))
            model_path = file_util.resolve_subpath(file_structure.get_result_folder(global_parameters),
                                                       'randomforest.pkl.gz')
            model = joblib.load(model_path)
            logger.log('Predicting data')
            batch_size = local_parameters['batch_size']
            if batch_size is None:
                batch_size = len(preprocessed)
            with progressbar.ProgressBar(len(preprocessed)) as progress:
                for i in range(int(math.ceil(len(preprocessed) / batch_size))):
                    start = i * batch_size
                    end = min(len(preprocessed), (i + 1) * batch_size)
                    results = random_forest.predict(preprocessed[start:end], model)
                    print('src shape: ' + str(results.shape))
                    print('dest shape: ' + str(predictions[start:end].shape))
                    predictions[start:end] = results[:]
                    progress.increment(end - start)
            preprocessed_h5.close()
            prediction_h5.close()
            file_util.move_file(temp_prediction_path, prediction_path)
