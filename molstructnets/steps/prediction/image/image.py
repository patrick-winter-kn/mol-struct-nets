import math

import h5py
from keras import models

from util import data_validation, file_structure, logger, file_util, progressbar, images, constants, hdf5_util


class Image:

    @staticmethod
    def get_id():
        return 'image'

    @staticmethod
    def get_name():
        return 'Image'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'batch_size', 'name': 'Batch Size', 'type': int, 'default': 1, 'min': 1,
                           'description': 'Number of data points that will be processed together. A higher number leads'
                                          ' to faster processing but needs more memory. Default: 1'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed_images(global_parameters)
        data_validation.validate_network(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        prediction_path = file_structure.get_prediction_file(global_parameters)
        if file_util.file_exists(prediction_path):
            logger.log('Skipping step: ' + prediction_path + ' already exists')
        else:
            model_path = file_structure.get_network_file(global_parameters)
            model = models.load_model(model_path)
            n = global_parameters[constants.GlobalParameters.n]
            temp_prediction_path = file_util.get_temporary_file_path('image_prediction')
            prediction_h5 = h5py.File(temp_prediction_path, 'w')
            predictions = hdf5_util.create_dataset(prediction_h5, file_structure.Predictions.prediction, (n, 2))
            logger.log('Predicting data')
            with progressbar.ProgressBar(n) as progress:
                for i in range(int(math.ceil(n / local_parameters['batch_size']))):
                    start = i * local_parameters['batch_size']
                    end = min(n, (i + 1) * local_parameters['batch_size'])
                    img_array = images.load_images(global_parameters[constants.GlobalParameters.preprocessed_data],
                                                   global_parameters[constants.GlobalParameters.input_dimensions][0],
                                                   global_parameters[constants.GlobalParameters.input_dimensions][1],
                                                   start, end)
                    results = model.predict(img_array)
                    predictions[start:end] = results[:]
                    progress.increment(len(img_array))
            prediction_h5.close()
            file_util.move_file(temp_prediction_path, prediction_path)
