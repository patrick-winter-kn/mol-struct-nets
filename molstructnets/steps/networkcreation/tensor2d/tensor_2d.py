from keras import initializers, optimizers
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Model

from util import file_structure, file_util, logger, constants


class Tensor2D:

    @staticmethod
    def get_id():
        return 'tensor_2d'

    @staticmethod
    def get_name():
        return '2D Tensor'

    @staticmethod
    def get_parameters():
        parameters = list()
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        dimensions = global_parameters[constants.GlobalParameters.input_dimensions]
        if len(dimensions) != 3:
            raise ValueError('Preprocessed dimensions are not 2D')

    @staticmethod
    def execute(global_parameters, local_parameters):
        network_path = file_structure.get_network_file(global_parameters)
        if file_util.file_exists(network_path):
            logger.log('Skipping step: ' + network_path + ' already exists')
        else:
            initializer = initializers.he_uniform()
            input_layer = Input(shape=global_parameters[constants.GlobalParameters.input_dimensions], name='input')
            layer = input_layer
            layer = Dropout(0.3, name='input-dropout')(layer)

            # Block 1
            layer = Convolution2D(32, 3, activation='relu', padding='same', name='convolution-1',
                                  kernel_initializer=initializer)(layer)
            layer = MaxPooling2D(2, padding='same', name='max-pool-1')(layer)

            # Block 2
            layer = Convolution2D(64, 3, activation='relu', padding='same', name='convolution-2',
                                  kernel_initializer=initializer)(layer)
            layer = MaxPooling2D(2, padding='same', name='max-pool-2')(layer)

            # Block 3
            layer = Convolution2D(128, 3, activation='relu', padding='same', name='convolution-3',
                                  kernel_initializer=initializer)(layer)
            layer = MaxPooling2D(2, padding='same', name='max-pool-3')(layer)

            # Block 4
            layer = Convolution2D(256, 3, activation='relu', padding='same', name='convolution-4',
                                  kernel_initializer=initializer)(layer)
            layer = MaxPooling2D(2, padding='same', name='max-pool-4')(layer)

            # Block 5
            layer = Convolution2D(512, 3, activation='relu', padding='same', name='convolution-5',
                                  kernel_initializer=initializer)(layer)
            layer = MaxPooling2D(2, padding='same', name='max-pool-5')(layer)

            layer = Flatten(name='features')(layer)
            layer = Dropout(0.75, name='dropout-1')(layer)
            layer = Dense(128, activation='relu', name='dense', kernel_initializer=initializer)(layer)
            layer = Dropout(0.75, name='dropout-2')(layer)
            output_layer = Dense(2, activation='softmax', name='output', kernel_initializer=initializer)(layer)
            model = Model(inputs=input_layer, outputs=output_layer)
            optimizer = optimizers.Adam(lr=0.0001)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
            file_util.make_folders(network_path)
            model.save(network_path)
