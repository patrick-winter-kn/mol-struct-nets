from util import data_validation, file_structure, reference_data_set, hdf5_util, logger, callbacks, constants
import h5py
from keras import models
from keras.callbacks import ModelCheckpoint, TensorBoard


class Matrix:

    @staticmethod
    def get_id():
        return 'matrix'

    @staticmethod
    def get_name():
        return 'Matrix'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'epochs', 'name': 'Epochs', 'type': int,
                           'description': 'The number of times the model will be trained on the whole data set.'})
        parameters.append({'id': 'batch_size', 'name': 'Batch size (default: 50)', 'type': int, 'default': 50,
                           'description': 'Number of data points that will be processed together. A higher number leads'
                                          ' to faster processing but needs more memory.'})
        parameters.append({'id': 'validation', 'name': 'Validation', 'type': bool, 'default': False,
                           'description': 'Evaluate the model after each epoch using the test data set.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed(global_parameters)
        data_validation.validate_network(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        # TODO validation
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
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            partition_h5 = h5py.File(global_parameters[constants.GlobalParameters.partition_data], 'r')
            train = partition_h5[file_structure.Partitions.train]
            input_ = reference_data_set.ReferenceDataSet(train, preprocessed)
            output = reference_data_set.ReferenceDataSet(train, classes)
            model = models.load_model(model_path)
            checkpoint = ModelCheckpoint(model_path)
            tensorboard = TensorBoard(log_dir=model_path[:-3] + '-tensorboard', histogram_freq=1, write_graph=True,
                                      write_images=False, embeddings_freq=1)
            custom_checkpoint = callbacks.CustomCheckpoint(model_path)
            model.fit(input_, output, epochs=local_parameters['epochs'], shuffle='batch',
                      batch_size=local_parameters['batch_size'], callbacks=[checkpoint, custom_checkpoint, tensorboard],
                      initial_epoch=epoch)
            target_h5.close()
            preprocessed_h5.close()
            partition_h5.close()
