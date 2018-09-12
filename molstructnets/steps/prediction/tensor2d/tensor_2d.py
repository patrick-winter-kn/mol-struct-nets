import h5py
from keras import models
import numpy

from steps.prediction.shared.tensor2d import prediction_array
from util import data_validation, file_structure, progressbar, logger, file_util, hdf5_util, misc


class Tensor2D:

    @staticmethod
    def get_id():
        return 'tensor_2d'

    @staticmethod
    def get_name():
        return 'Tensor 2D'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'batch_size', 'name': 'Batch Size', 'type': int, 'default': 100, 'min': 1,
                           'description': 'Number of data points that will be processed together. A higher number leads'
                                          ' to faster processing but needs more memory. Default: 100'})
        parameters.append({'id': 'number_predictions', 'name': 'Predictions per data point', 'type': int, 'default': 1,
                           'min': 1, 'description': 'The number of times a data point is predicted (with different'
                                                    ' transformations). The result is the mean of all predictions. Default: 1'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_preprocessed_specs(global_parameters)
        data_validation.validate_network(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        prediction_path = file_structure.get_prediction_file(global_parameters)
        if file_util.file_exists(prediction_path):
            logger.log('Skipping step: ' + prediction_path + ' already exists')
        else:
            array = prediction_array.PredictionArrays(global_parameters, local_parameters['batch_size'],
                                                      transformations=local_parameters['number_predictions'])
            predictions = numpy.zeros((len(array.input), 2))
            temp_prediction_path = file_util.get_temporary_file_path('tensor_prediction')
            model_path = file_structure.get_network_file(global_parameters)
            model = models.load_model(model_path)
            logger.log('Predicting data')
            chunks = misc.chunk_by_size(len(array.input), local_parameters['batch_size'])
            with progressbar.ProgressBar(len(array.input) * local_parameters['number_predictions']) as progress:
                for iteration in range(local_parameters['number_predictions']):
                    for chunk in chunks:
                        predictions[chunk['start']:chunk['end']] += model.predict(array.input.next())[:]
                        progress.increment(chunk['size'])
            predictions /= local_parameters['number_predictions']
            array.close()
            prediction_h5 = h5py.File(temp_prediction_path, 'w')
            hdf5_util.create_dataset_from_data(prediction_h5, file_structure.Predictions.prediction, predictions)
            prediction_h5.close()
            file_util.move_file(temp_prediction_path, prediction_path)
