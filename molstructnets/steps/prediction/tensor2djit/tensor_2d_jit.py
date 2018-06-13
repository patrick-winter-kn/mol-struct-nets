from util import data_validation, file_structure, progressbar, logger, file_util, hdf5_util, misc, thread_pool
from keras import models
import h5py
from steps.preprocessing.shared.tensor2d import tensor_2d_jit_array
import queue


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
        parameters.append({'id': 'number_predictions', 'name': 'Predictions per data point', 'type': int, 'default': 1,
                           'min': 1, 'description': 'The number of times a data point is predicted (with different'
                                          ' transformations). The result is the mean of all predictions. Default: 1'})
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
            multiple = local_parameters['number_predictions'] > 1
            array = tensor_2d_jit_array.load_array(global_parameters, transform=multiple)
            data_queue = queue.Queue(10)
            temp_prediction_path = file_util.get_temporary_file_path('tensor_prediction')
            prediction_h5 = h5py.File(temp_prediction_path, 'w')
            predictions = hdf5_util.create_dataset(prediction_h5, file_structure.Predictions.prediction,
                                                   (len(array), 2))
            model_path = file_structure.get_network_file(global_parameters)
            model = models.load_model(model_path)
            logger.log('Predicting data')
            chunks = misc.chunk_by_size(len(array), local_parameters['batch_size'])
            with thread_pool.ThreadPool(1) as pool:
                pool.submit(generate_data, array, chunks, local_parameters['number_predictions'], data_queue)
                with progressbar.ProgressBar(len(array)) as progress:
                    for chunk in chunks:
                        predictions_chunk = model.predict(data_queue.get())
                        if multiple:
                            for i in range(1, local_parameters['number_predictions']):
                                predictions_chunk += model.predict(data_queue.get())
                            predictions_chunk /= local_parameters['number_predictions']
                        predictions[chunk['start']:chunk['end']+1] = predictions_chunk[:]
                        progress.increment(chunk['end'] + 1 - chunk['start'])
            array.close()
            prediction_h5.close()
            file_util.move_file(temp_prediction_path, prediction_path)


def generate_data(array, chunks, number_predictions, data_queue):
    for chunk in chunks:
        data_queue.put(array[chunk['start']:chunk['end']+1])
        if number_predictions > 1:
            for i in range(1, number_predictions):
                array.set_iteration(i)
                data_queue.put(array[chunk['start']:chunk['end']+1])
            array.set_iteration(0)
