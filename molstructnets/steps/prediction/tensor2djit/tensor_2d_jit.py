from util import data_validation, file_structure, progressbar, logger, file_util, hdf5_util, misc
from keras import models
import h5py
import math
from steps.preprocessing.shared.tensor2d import tensor_2d_jit_array


class Tensor2DJit:

    @staticmethod
    def get_id():
        return 'tensor_2d_jit'

    @staticmethod
    def get_name():
        return 'Tensor 2D JIT'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'batch_size', 'name': 'Batch Size', 'type': int, 'default': 50, 'min': 1,
                           'description': 'Number of data points that will be processed together. A higher number leads'
                                          ' to faster processing but needs more memory. Default: 50'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_preprocessed_jit(global_parameters)
        data_validation.validate_network(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        prediction_path = file_structure.get_prediction_file(global_parameters)
        if file_util.file_exists(prediction_path):
            logger.log('Skipping step: ' + prediction_path + ' already exists')
        else:
            array = tensor_2d_jit_array.load_array(global_parameters)
            temp_prediction_path = file_util.get_temporary_file_path('tensor_prediction')
            prediction_h5 = h5py.File(temp_prediction_path, 'w')
            predictions = hdf5_util.create_dataset(prediction_h5, file_structure.Predictions.prediction,
                                                   (len(array), 2))
            model_path = file_structure.get_network_file(global_parameters)
            model = models.load_model(model_path)
            logger.log('Predicting data')
            chunks = misc.chunk_by_size(len(array), local_parameters['batch_size'])
            with progressbar.ProgressBar(len(array)) as progress:
                for chunk in chunks:
                    predictions[chunk['start']:chunk['end']+1] = model.predict(array[chunk['start']:chunk['end']+1])[:]
                    progress.increment(chunk['end'] + 1 - chunk['start'])
            array.close()
            prediction_h5.close()
            file_util.move_file(temp_prediction_path, prediction_path)
