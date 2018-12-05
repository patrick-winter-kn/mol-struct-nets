from keras import models
import numpy

from steps.training.tensor2d import training_array
from keras.callbacks import Callback
from util import data_validation, file_structure, logger, callbacks, file_util, misc, progressbar, constants,\
    process_pool
from steps.evaluation.shared import enrichment, roc_curve
from steps.prediction.shared.tensor2d import prediction_array


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
            epochs = local_parameters['epochs']
            batch_size = local_parameters['batch_size']
            model = models.load_model(model_path)
            process_pool_ = process_pool.ProcessPool()
            arrays = training_array.TrainingArrays(global_parameters, epochs - epoch, epoch, batch_size, multi_process=process_pool_)
            callbacks_ = [callbacks.CustomCheckpoint(model_path)]
            test_data = None
            if local_parameters['evaluate']:
                test_data = prediction_array.PredictionArrays(global_parameters, local_parameters['batch_size'],
                                                              test=True, runs=epochs - epoch, multi_process=process_pool_,
                                                              percent=local_parameters['eval_partition_size'] * 0.01)
                chunks = misc.chunk_by_size(len(test_data.input), local_parameters['batch_size'])
                callbacks_ = [EvaluationCallback(test_data, chunks, global_parameters[constants.GlobalParameters.seed],
                                                 model_path[:-3] + '-eval.csv')] + callbacks_
            model.fit(arrays.input, arrays.output, epochs=epochs, shuffle=False, batch_size=batch_size,
                      callbacks=callbacks_, initial_epoch=epoch)
            if test_data is not None:
                test_data.close()
            arrays.close()
            process_pool_.close()


class EvaluationCallback(Callback):

    def __init__(self, test_data, chunks, seed, file_path):
        self.test_data = test_data
        self.seed = seed
        self.file_path = file_path
        self.chunks = chunks

    def on_epoch_end(self, epoch, logs=None):
        predictions = numpy.zeros((len(self.test_data.input), 2))
        with progressbar.ProgressBar(len(self.test_data.input)) as progress:
            for chunk in self.chunks:
                predictions_chunk = self.model.predict(self.test_data.input.next())
                predictions[chunk['start']:chunk['end']] = predictions_chunk[:]
                progress.increment(chunk['size'])
        actives, e_auc, efs = enrichment.stats(predictions, self.test_data.output, [5, 10], seed=self.seed)
        roc_auc = roc_curve.stats(predictions, self.test_data.output, seed=self.seed)[2]
        ef5 = efs[5]
        ef10 = efs[10]
        write_headline = not file_util.file_exists(self.file_path)
        with open(self.file_path, 'a') as file:
            if write_headline:
                file.write('epoch,e_auc,ef_5,ef_10,roc_auc\n')
            file.write(str(epoch + 1) + ',' + str(e_auc) + ',' + str(ef5) + ',' + str(ef10) + ',' + str(roc_auc) + '\n')
        logger.log('E AUC: ' + str(round(e_auc, 2)) + '    EF5: ' + str(round(ef5, 2)) + '    EF10: '
                   + str(round(ef10, 2)) + '    ROC AUC: ' + str(round(roc_auc, 2)))
