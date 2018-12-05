from keras import models

from steps.trainingmultitarget.tensor2d import multitarget_training_array
from steps.training.shared.tensor2d import weight_transfer
from util import file_structure, logger, callbacks, file_util, constants, progressbar


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
        parameters.append({'id': 'frozen_runs', 'name': 'Frozen Runs', 'type': int, 'default': 1, 'min': 0,
                           'description': 'Number of times the network will be trained with frozen feature layers '
                                          'before they are unfrozen and the real training run is started. Default: 1'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        pass

    @staticmethod
    def execute(global_parameters, local_parameters):
        model_path = file_structure.get_network_file(global_parameters, True)
        epoch_path = model_path[:-3] + '-epochs.txt'
        if file_util.file_exists(epoch_path):
            with open(epoch_path, 'r') as file:
                epoch = int(file.readline())
        else:
            epoch = 0
        if epoch >= local_parameters['epochs']:
            logger.log('Skipping step: ' + model_path + ' has already been trained for ' + str(epoch) + ' epochs')
        else:
            data_sets = global_parameters[constants.GlobalParameters.data_set]
            epochs = local_parameters['epochs']
            batch_size = local_parameters['batch_size']
            frozen_runs = local_parameters['frozen_runs']
            arrays = multitarget_training_array.MultitargetTrainingArrays(global_parameters, epochs - epoch, epoch,
                                                                          batch_size, frozen_runs)
            shared_model = models.load_model(model_path)
            model_list = list()
            for i in range(len(data_sets)):
                new_model = models.clone_model(shared_model)
                new_model.compile(optimizer=shared_model.optimizer, loss=shared_model.loss, metrics=shared_model.metrics)
                model_list.append(new_model)
            layer_start_index, layer_end_index = weight_transfer.get_layer_range(shared_model, 'input', 'features')
            weight_start_index, weight_end_index = weight_transfer.get_weight_range(shared_model, 'input', 'features')
            logger.log('Running multitarget training with:\nEpochs: ' + str(epochs) + '\nData sets: '
                       + str(len(data_sets)) + '\nBatches per epoch: ' + str(arrays.batches_per_epoch())
                       + '\nBatch size: ' + str(batch_size) + '\nFrozen runs: ' + str(frozen_runs))
            with progressbar.ProgressBar(epochs * arrays.batches_per_epoch() * len(data_sets) * batch_size) as progress:
                progress.increment(epoch * arrays.batches_per_epoch() * len(data_sets) * batch_size)
                prev_model = shared_model
                for i in range(epoch, epochs):
                    for j in range(arrays.batches_per_epoch()):
                        for k in range(len(data_sets)):
                            weight_transfer.transfer_weights(prev_model, model_list[k], weight_start_index, weight_end_index)
                            if frozen_runs > 0:
                                weight_transfer.set_weight_freeze(model_list[k], layer_start_index, layer_end_index, True)
                                model_list[k].fit(arrays.input, arrays.output, epochs=frozen_runs, shuffle=False,
                                               batch_size=batch_size, verbose=0)
                                weight_transfer.set_weight_freeze(model_list[k], layer_start_index, layer_end_index, False)
                            model_list[k].fit(arrays.input, arrays.output, epochs=1, shuffle=False, batch_size=batch_size, verbose=0)
                            progress.increment(batch_size)
                            prev_model = model_list[k]
                        weight_transfer.transfer_weights(prev_model, shared_model, weight_start_index, weight_end_index)
                    callbacks.save_model(shared_model, model_path, i)
            arrays.close()
