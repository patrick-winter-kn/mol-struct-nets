from keras import models
import queue
import numpy

from steps.training.tensor2d import training_array
from keras.callbacks import Callback
from util import data_validation, file_structure, logger, callbacks, file_util, misc, thread_pool, progressbar,\
    constants, process_pool
from steps.evaluation.shared import enrichment
from steps.preprocessing.shared.tensor2d import tensor_2d_array


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
        parameters.append({'id': 'epochs', 'name': 'Epochs', 'type': int, 'min': 1,
                           'description': 'The number of times the model will be trained on the whole data set.'})
        parameters.append({'id': 'batch_size', 'name': 'Batch Size', 'type': int, 'default': 100, 'min': 1,
                           'description': 'Number of data points that will be processed together. A higher number leads'
                                          ' to faster processing but needs more memory. Default: 100'})
        parameters.append({'id': 'evaluate', 'name': 'Evaluate', 'type': bool, 'default': False,
                           'description': 'Evaluate on the test data after each epoch. Default: False'})
        parameters.append({'id': 'eval_partition_size', 'name': 'Evaluation partition size', 'type': int,
                           'default': 100, 'min': 1, 'max': 100, 'description':
                               'The size in percent of the test partition used for evaluation. Default: 100'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed_specs(global_parameters)
        data_validation.validate_network(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        model_path = file_structure.get_network_file(global_parameters)
        epoch_path = model_path[:-3] + '-epochs.txt'
        if file_util.file_exists(epoch_path):
            with open(epoch_path, 'r') as file:
                epoch = int(file.read())
        else:
            epoch = 0
        if epoch >= local_parameters['epochs']:
            logger.log('Skipping step: ' + model_path + ' has already been trained for ' + str(epoch) + ' epochs')
        else:
            epochs = local_parameters['epochs']
            batch_size = local_parameters['batch_size']
            model = models.load_model(model_path)
            process_pool_ = process_pool.ProcessPool()
            arrays = training_array.TrainingArrays(global_parameters, epochs, batch_size, multi_process=process_pool_)
            callbacks_ = [callbacks.CustomCheckpoint(model_path)]
            pool = None
            if local_parameters['evaluate']:
                test_data = tensor_2d_array.load_array(global_parameters, test=True, multi_process=process_pool_,
                                                       percent=local_parameters['eval_partition_size'] * 0.01)
                pool = thread_pool.ThreadPool(1)
                chunks = misc.chunk_by_size(len(test_data), local_parameters['batch_size'])
                data_queue = queue.Queue(10)
                pool.submit(generate_data, test_data, chunks, epochs - epoch, data_queue)
                callbacks_ = [EvaluationCallback(test_data, chunks, data_queue,
                                                 global_parameters[constants.GlobalParameters.seed],
                                                 model_path[:-3] + '-eval.txt')] + callbacks_
            model.fit(arrays.input, arrays.output, epochs=epochs, shuffle=False, batch_size=batch_size,
                      callbacks=callbacks_, initial_epoch=epoch)
            if pool is not None:
                pool.close()
            arrays.close()


class EvaluationCallback(Callback):

    def __init__(self, test_data, chunks, data_queue, seed, file_path):
        self.test_data = test_data
        self.seed = seed
        self.file_path = file_path
        self.data_queue = data_queue
        self.chunks = chunks

    def on_epoch_end(self, epoch, logs=None):
        predictions = numpy.zeros((len(self.test_data), 2))
        with progressbar.ProgressBar(len(self.test_data)) as progress:
            for chunk in self.chunks:
                predictions_chunk = self.model.predict(self.data_queue.get())
                predictions[chunk['start']:chunk['end']] = predictions_chunk[:]
                progress.increment(chunk['size'])
        actives, auc, efs = enrichment.stats(predictions, self.test_data.classes(), [5], shuffle=True, seed=self.seed)
        ef5 = efs[5]
        with open(self.file_path, 'a') as file:
            file.write(str(epoch + 1) + ',' + str(auc) + ',' + str(ef5) + '\n')
        logger.log('AUC: ' + str(round(auc, 2)) + '    EF5: ' + str(round(ef5, 2)))


def generate_data(array, chunks, epochs, data_queue):
    for i in range(epochs):
        for chunk in chunks:
            data_queue.put(array[chunk['start']:chunk['end']])
