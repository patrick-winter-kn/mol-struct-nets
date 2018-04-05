from util import data_validation, file_structure, progressbar, logger, file_util, constants, hdf5_util
from keras import models
import h5py
import math


class Tensor:

    @staticmethod
    def get_id():
        return 'tensor'

    @staticmethod
    def get_name():
        return 'Tensor'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'batch_size', 'name': 'Batch Size', 'type': int, 'default': 50, 'min': 1,
                           'description': 'Number of data points that will be processed together. A higher number leads'
                                          ' to faster processing but needs more memory. Default: 50'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_preprocessed(global_parameters)
        data_validation.validate_network(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        prediction_path = file_structure.get_prediction_file(global_parameters)
        if file_util.file_exists(prediction_path):
            logger.log('Skipping step: ' + prediction_path + ' already exists')
        else:
            preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data], 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            temp_prediction_path = file_util.get_temporary_file_path('tensor_prediction')
            prediction_h5 = h5py.File(temp_prediction_path, 'w')
            predictions = hdf5_util.create_dataset(prediction_h5, file_structure.Predictions.prediction,
                                                   (len(preprocessed), 2))
            model_path = file_structure.get_network_file(global_parameters)
            model = models.load_model(model_path)
            logger.log('Predicting data')
            with progressbar.ProgressBar(len(preprocessed)) as progress:
                for i in range(int(math.ceil(len(preprocessed) / local_parameters['batch_size']))):
                    start = i * local_parameters['batch_size']
                    end = min(len(preprocessed), (i + 1) * local_parameters['batch_size'])
                    results = model.predict(preprocessed[start:end])
                    predictions[start:end] = results[:]
                    progress.increment(end - start)
            preprocessed_h5.close()
            prediction_h5.close()
            file_util.move_file(temp_prediction_path, prediction_path)
