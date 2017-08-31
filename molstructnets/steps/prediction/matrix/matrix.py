from util import data_validation, file_structure, progressbar, logger, file_util
from keras import models
import h5py
import math


class Matrix:

    @staticmethod
    def get_id():
        return 'matrix'

    @staticmethod
    def get_name():
        return 'Matrix'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'batch_size', 'name': 'Batch size', 'type': int, 'default': 50})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed(global_parameters)
        data_validation.validate_network(global_parameters)

    @staticmethod
    def execute(global_parameters, parameters):
        preprocessed_path = global_parameters['preprocessed_data']
        if file_util.file_exists(preprocessed_path):
            logger.log('Skipping step: ' + preprocessed_path + ' already exists')
        else:
            preprocessed_h5 = h5py.File(preprocessed_path, 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            prediction_path = file_structure.get_prediction_file(global_parameters)
            temp_prediction_path = file_util.get_temporary_file_path('matrix_prediction')
            prediction_h5 = h5py.File(temp_prediction_path, 'w')
            predictions = prediction_h5.create_dataset(file_structure.Predictions.prediction, (len(preprocessed), 2))
            model_path = file_structure.get_network_file(global_parameters)
            model = models.load_model(model_path)
            logger.log('Predicting training data')
            with progressbar.ProgressBar(len(preprocessed)) as progress:
                for i in range(int(math.ceil(len(preprocessed) / parameters['batch_size']))):
                    start = i * parameters['batch_size']
                    end = min(len(preprocessed), (i + 1) * parameters['batch_size'])
                    results = model.predict(preprocessed[start:end])
                    predictions[start:end] = results[:]
                    progress.update(end - start)
            preprocessed_h5.close()
            prediction_h5.close()
            file_util.move_file(temp_prediction_path, prediction_path)
