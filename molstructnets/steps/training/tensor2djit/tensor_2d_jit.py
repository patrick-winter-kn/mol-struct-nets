from keras import models

from steps.training.tensor2djit import training_array
from util import data_validation, file_structure, logger, callbacks, file_util


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
        parameters.append({'id': 'batch_size', 'name': 'Batch Size', 'type': int, 'default': 100, 'min': 1,
                           'description': 'Number of data points that will be processed together. A higher number leads'
                                          ' to faster processing but needs more memory. Default: 100'})
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
            epochs = local_parameters['epochs']
            batch_size = local_parameters['batch_size']
            model = models.load_model(model_path)
            arrays = training_array.TrainingArrays(global_parameters, epochs, batch_size)
            model.fit(arrays.input, arrays.output, epochs=epochs, shuffle=False, batch_size=batch_size,
                      callbacks=[callbacks.CustomCheckpoint(model_path)], initial_epoch=epoch)
            arrays.close()
