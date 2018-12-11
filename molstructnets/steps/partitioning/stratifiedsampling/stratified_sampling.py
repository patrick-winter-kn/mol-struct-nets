import random

import h5py
import numpy

from steps.partitioning.shared import partitioning
from util import data_validation, file_structure, misc, file_util, progressbar, logger, constants, hdf5_util


class StratifiedSampling:

    @staticmethod
    def get_id():
        return 'stratified_sampling'

    @staticmethod
    def get_name():
        return 'Stratified Sampling'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'train_percentage', 'name': 'Size of Training Partition', 'type': int, 'min': 1,
                           'max': 100,
                           'description': 'The percentage of the data that will be used for training.'})
        parameters.append({'id': 'oversample', 'name': 'Oversample Training Partition', 'type': bool, 'default': True,
                           'description': 'If this is set the minority class will be oversampled, so that the class'
                                          ' distribution in the training set is equal. Default: True'})
        parameters.append({'id': 'shuffle', 'name': 'Shuffle Training Partition', 'type': bool, 'default': True,
                           'description': 'If this is set the training data will be shuffled. Default: True'})
        parameters.append({'id': 'seed', 'name': 'Random Seed', 'type': int, 'default': None,
                           'description': 'The used random seed. Default: Use global seed'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        if local_parameters['seed'] is None:
            hash_parameters = misc.copy_dict_from_keys(global_parameters, [constants.GlobalParameters.seed])
        else:
            hash_parameters = dict()
        hash_parameters.update(misc.copy_dict_from_keys(local_parameters, ['train_percentage', 'oversample',
                                                                           'shuffle', 'seed']))
        file_name = 'stratified_sampling_' + misc.hash_parameters(hash_parameters) + '.h5'
        return file_util.resolve_subpath(file_structure.get_partition_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        partition_path = StratifiedSampling.get_result_file(global_parameters, local_parameters)
        if constants.GlobalParameters.partition_data in global_parameters:
            logger.log('Partitioning has already been specified. Overwriting partition parameter with generated'
                       ' partitions.', logger.LogLevel.WARNING)
        global_parameters[constants.GlobalParameters.partition_data] = file_util.get_filename(partition_path,
                                                                                              with_extension=False)
        if file_util.file_exists(partition_path):
            logger.log('Skipping step: ' + partition_path + ' already exists')
        else:
            if local_parameters['seed'] is None:
                seed = global_parameters[constants.GlobalParameters.seed]
            else:
                seed = local_parameters['seed']
            random_ = random.Random(seed)
            target_h5 = h5py.File(file_structure.get_target_file(global_parameters), 'r')
            classes = target_h5[file_structure.Target.classes]
            classes = classes[:].astype('bool')
            temp_partition_path = file_util.get_temporary_file_path('stratified_sampling')
            partition_h5 = h5py.File(temp_partition_path, 'w')
            # Get list of indices for not zero elements in first/second column (actives/inacitves)
            active_indices = list((classes[:, 0].nonzero()[0]).astype('int32'))
            inactive_indices = list((classes[:, 1].nonzero()[0]).astype('int32'))
            logger.log('Found ' + str(len(active_indices)) + ' active indices and ' + str(len(inactive_indices)) +
                       ' inactive data points', logger.LogLevel.VERBOSE)
            number_training = round(len(classes) * local_parameters['train_percentage'] * 0.01)
            number_training_active = round(number_training * (len(active_indices) / len(classes)))
            number_training_inactive = number_training - number_training_active
            logger.log('Picking data points for training', logger.LogLevel.VERBOSE)
            with progressbar.ProgressBar(number_training, logger.LogLevel.VERBOSE) as progress:
                for i in range(number_training_active):
                    del active_indices[random_.randint(0, len(active_indices) - 1)]
                    progress.increment()
                for i in range(number_training_inactive):
                    del inactive_indices[random_.randint(0, len(inactive_indices) - 1)]
                    progress.increment()
            partition_train = numpy.zeros(number_training, dtype='uint32')
            partition_test = numpy.zeros(classes.shape[0] - number_training, dtype='uint32')
            logger.log('Writing partitions', logger.LogLevel.VERBOSE)
            # Convert actives and inactives into set to speed up processing
            actives = set(active_indices)
            inactives = set(inactive_indices)
            with progressbar.ProgressBar(len(classes), logger.LogLevel.VERBOSE) as progress:
                partition_train_index = 0
                partition_test_index = 0
                for i in range(classes.shape[0]):
                    if classes[i, 0] > 0.0:
                        if i in actives:
                            partition_test[partition_test_index] = i
                            partition_test_index += 1
                        else:
                            partition_train[partition_train_index] = i
                            partition_train_index += 1
                    else:
                        if i in inactives:
                            partition_test[partition_test_index] = i
                            partition_test_index += 1
                        else:
                            partition_train[partition_train_index] = i
                            partition_train_index += 1
                    progress.increment()
            if local_parameters['oversample']:
                partition_train = partitioning.oversample(partition_train, classes, log_level=logger.LogLevel.VERBOSE)
            if local_parameters['shuffle']:
                numpy.random.seed(seed)
                numpy.random.shuffle(partition_train)
            hdf5_util.create_dataset_from_data(partition_h5, file_structure.Partitions.train, partition_train)
            hdf5_util.create_dataset_from_data(partition_h5, file_structure.Partitions.test, partition_test)
            target_h5.close()
            partition_h5.close()
            hdf5_util.set_property(temp_partition_path, 'train_percentage', local_parameters['train_percentage'])
            hdf5_util.set_property(temp_partition_path, 'oversample', local_parameters['oversample'])
            hdf5_util.set_property(temp_partition_path, 'shuffle', local_parameters['shuffle'])
            file_util.move_file(temp_partition_path, partition_path)
