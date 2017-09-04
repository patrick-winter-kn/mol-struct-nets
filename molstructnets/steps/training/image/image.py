from util import data_validation, file_structure, hdf5_util, logger, reference_data_set, callbacks, images, constants
from keras import models
from keras.callbacks import ModelCheckpoint
import h5py


class Image:

    @staticmethod
    def get_id():
        return 'image'

    @staticmethod
    def get_name():
        return 'Image'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'epochs', 'name': 'Epochs', 'type': int,
                           'description': 'The number of times the model will be trained on the whole data set.'})
        parameters.append({'id': 'batch_size', 'name': 'Batch size (default: 1)', 'type': int, 'default': 1,
                           'description': 'Number of data points that will be processed together. A higher number leads'
                                          ' to faster processing but needs more memory.'})
        parameters.append({'id': 'validation', 'name': 'Validation', 'type': bool, 'default': False,
                           'description': 'Evaluate the model after each epoch using the test data set.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed_images(global_parameters)
        data_validation.validate_network(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        # TODO more callbacks?
        model_path = file_structure.get_network_file(global_parameters)
        epoch = hdf5_util.get_property(model_path, 'epochs_trained')
        if epoch is None:
            epoch = 0
        if epoch >= local_parameters['epochs']:
            logger.log('Skipping step: ' + model_path + ' has already been trained for ' + str(epoch) + ' epochs')
        else:
            target_h5 = h5py.File(file_structure.get_target_file(global_parameters), 'r')
            classes = target_h5[file_structure.Target.classes]
            partition_h5 = h5py.File(global_parameters[constants.GlobalParameters.partition_data], 'r')
            train = partition_h5[file_structure.Partitions.train]
            logger.log('Loading images')
            img_array = images.load_images(global_parameters[constants.GlobalParameters.preprocessed_data],
                                           global_parameters[constants.GlobalParameters.input_dimensions][0],
                                           global_parameters[constants.GlobalParameters.input_dimensions][1], 0,
                                           len(train), True)
            output = reference_data_set.ReferenceDataSet(train, classes)
            model = models.load_model(model_path)
            checkpoint = ModelCheckpoint(model_path)
            custom_checkpoint = callbacks.CustomCheckpoint(model_path)
            model.fit(img_array, output, epochs=local_parameters['epochs'], shuffle='batch',
                      batch_size=local_parameters['batch_size'], callbacks=[checkpoint, custom_checkpoint],
                      initial_epoch=epoch)
            target_h5.close()
            partition_h5.close()
