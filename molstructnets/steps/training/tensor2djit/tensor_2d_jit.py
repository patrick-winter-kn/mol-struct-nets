from keras import models

from util import data_validation, file_structure, logger, callbacks, process_pool, file_util
from steps.preprocessing.shared.tensor2d import tensor_2d_jit_array
from steps.training.tensor2djit import tensor_2d_jit_data_generator


use_keras_workers = False


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
        parameters.append({'id': 'epochs', 'name': 'Epochs', 'type': int, 'min': 1,
                           'description': 'The number of times the model will be trained on the whole data set.'})
        parameters.append({'id': 'batch_size', 'name': 'Batch Size', 'type': int, 'default': 50, 'min': 1,
                           'description': 'Number of data points that will be processed together. A higher number leads'
                                          ' to faster processing but needs more memory. Default: 50'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed_jit(global_parameters)
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
            batch_size = local_parameters['batch_size']
            array = tensor_2d_jit_array.load_array(global_parameters, train=True, transform=True,
                                                   multi_process=not use_keras_workers)
            model = models.load_model(model_path)
            number_batches = tensor_2d_jit_data_generator.number_chunks(array, batch_size)
            logger.log('Training on ' + str(number_batches) + ' batches with size ' + str(batch_size))
            if use_keras_workers:
                keras_workers = process_pool.default_number_processes
            else:
                keras_workers = 1
            model.fit_generator(tensor_2d_jit_data_generator.generate_data(array, batch_size), number_batches,
                                epochs=local_parameters['epochs'], callbacks=[callbacks.CustomCheckpoint(model_path)],
                                initial_epoch=epoch, workers=keras_workers)
            array.close()
