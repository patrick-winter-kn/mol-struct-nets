import math

import h5py
import numpy
from keras import models
from keras.callbacks import ModelCheckpoint, Callback

from steps.evaluation.shared import enrichment
from util import data_validation, file_structure, reference_data_set, hdf5_util, logger, callbacks, constants, \
    progressbar, misc


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
        parameters.append({'id': 'epochs', 'name': 'Epochs', 'type': int, 'min': 1,
                           'description': 'The number of times the model will be trained on the whole data set.'})
        parameters.append({'id': 'batch_size', 'name': 'Batch Size', 'type': int, 'default': 50, 'min': 1,
                           'description': 'Number of data points that will be processed together. A higher number leads'
                                          ' to faster processing but needs more memory. Default: 50'})
        parameters.append({'id': 'validation', 'name': 'Validation', 'type': bool, 'default': False,
                           'description': 'Evaluate the model after each epoch using the test data set. Default:'
                                          ' False'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed(global_parameters)
        data_validation.validate_network(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        model_path = file_structure.get_network_file(global_parameters)
        epoch = hdf5_util.get_property(model_path, 'epochs_trained')
        if epoch is None:
            epoch = 0
        if epoch >= local_parameters['epochs']:
            logger.log('Skipping step: ' + model_path + ' has already been trained for ' + str(epoch) + ' epochs')
        else:
            target_h5 = h5py.File(file_structure.get_target_file(global_parameters), 'r')
            classes = target_h5[file_structure.Target.classes]
            preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data], 'r')
            partition_h5 = h5py.File(file_structure.get_partition_file(global_parameters), 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            preprocessed_training_h5 = None
            only_boolean = len(preprocessed_h5[file_structure.Preprocessed.index]) == preprocessed.shape[-1]
            if constants.GlobalParameters.preprocessed_training_data in global_parameters:
                preprocessed_training_h5 = \
                    h5py.File(global_parameters[constants.GlobalParameters.preprocessed_training_data], 'r')
                train = preprocessed_training_h5[file_structure.PreprocessedTraining.preprocessed_training_references]
                input_ = preprocessed_training_h5[file_structure.PreprocessedTraining.preprocessed_training]
            else:
                train = partition_h5[file_structure.Partitions.train]
                input_ = reference_data_set.ReferenceDataSet(train, preprocessed)
            output = reference_data_set.ReferenceDataSet(train, classes)
            callback_list = list()
            if local_parameters['validation']:
                test = partition_h5[file_structure.Partitions.test]
                validation_input = reference_data_set.ReferenceDataSet(test, preprocessed)
                validation_output = reference_data_set.ReferenceDataSet(test, classes)
                validation_input = misc.copy_into_memory(validation_input, as_bool=only_boolean)
                validation_output = misc.copy_into_memory(validation_output, as_bool=True)
                callback_list.append(DrugDiscoveryEval(validation_input, validation_output,
                                                       local_parameters['batch_size']))
            model = models.load_model(model_path)
            callback_list.append(ModelCheckpoint(model_path))
            callback_list.append(callbacks.CustomCheckpoint(model_path))
            # callback_list.append(TensorBoard(log_dir=model_path[:-3] + '-tensorboard', histogram_freq=1,
            #                                 write_graph=True, write_images=False, embeddings_freq=1))
            input_ = misc.copy_into_memory(input_, as_bool=only_boolean)
            output = misc.copy_into_memory(output, as_bool=True)
            if isinstance(input, numpy.ndarray) and isinstance(output, numpy.ndarray):
                shuffle = True
            else:
                shuffle = 'batch'
            model.fit(input_, output, epochs=local_parameters['epochs'], shuffle=shuffle,
                      batch_size=local_parameters['batch_size'], callbacks=callback_list, initial_epoch=epoch)
            target_h5.close()
            preprocessed_h5.close()
            if preprocessed_training_h5 is not None:
                preprocessed_training_h5.close()
            if 'partition_h5' in locals():
                partition_h5.close()


class DrugDiscoveryEval(Callback):

    def __init__(self, input_, output, batch_size, ef_percent=None):
        super().__init__()
        if ef_percent is None:
            self.ef_percent = [5, 10]
        else:
            self.ef_percent = ef_percent
        self.input = input_
        self.output = output
        self.positives = enrichment.positives_count(output)
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        # Start with a new line, don't print right of the progressbar
        logger.log('\nPredicting with intermediate model')
        predictions = numpy.zeros((self.input.shape[0], 2))
        with progressbar.ProgressBar(len(self.input)) as progress:
            for i in range(int(math.ceil(len(self.input) / self.batch_size))):
                start = i * self.batch_size
                end = min(len(self.input), (i + 1) * self.batch_size)
                results = self.model.predict(self.input[start:end])
                predictions[start:end] = results[:]
                progress.increment(end - start)
        actives, auc, efs = enrichment.stats(predictions, self.output, self.ef_percent, positives=self.positives)
        for percent in sorted(efs.keys()):
            logs['enrichment_factor_' + str(percent)] = numpy.float64(efs[percent])
        logs['enrichment_auc'] = numpy.float64(auc)
        dr = self.diversity_ratio(predictions)
        logger.log('Diversity Ratio: ' + str(dr), logger.LogLevel.VERBOSE)
        logs['diversity_ratio'] = numpy.float64(dr)

    @staticmethod
    def diversity_ratio(predictions):
        results = set()
        for i in range(len(predictions)):
            results.add(predictions[i][0])
        return len(results) / len(predictions)
