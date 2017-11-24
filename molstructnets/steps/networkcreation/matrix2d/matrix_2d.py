from util import file_structure, file_util, logger, constants
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import initializers


class Matrix2D:

    @staticmethod
    def get_id():
        return 'matrix_2d'

    @staticmethod
    def get_name():
        return '2D Matrix'

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

            # Block 1
            layer = Convolution2D(16, 3, activation='relu', padding='same', name='convolution_1_1', kernel_initializer=initializer)(layer)
            layer = Convolution2D(16, 3, activation='relu', padding='same', name='convolution_1_2', kernel_initializer=initializer)(layer)
            layer = Convolution2D(16, 3, activation='relu', padding='same', name='convolution_1_3', kernel_initializer=initializer)(layer)
            layer = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_1')(layer)

            # Block 2
            layer = Convolution2D(32, 3, activation='relu', padding='same', name='convolution_2_1', kernel_initializer=initializer)(layer)
            layer = Convolution2D(32, 3, activation='relu', padding='same', name='convolution_2_2', kernel_initializer=initializer)(layer)
            layer = Convolution2D(32, 3, activation='relu', padding='same', name='convolution_2_3', kernel_initializer=initializer)(layer)
            layer = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_2')(layer)

            # Block 3
            layer = Convolution2D(64, 3, activation='relu', padding='same', name='convolution_3_1', kernel_initializer=initializer)(layer)
            layer = Convolution2D(64, 3, activation='relu', padding='same', name='convolution_3_2', kernel_initializer=initializer)(layer)
            layer = Convolution2D(64, 3, activation='relu', padding='same', name='convolution_3_3', kernel_initializer=initializer)(layer)
            layer = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_3')(layer)

            # Block 4
            layer = Convolution2D(128, 3, activation='relu', padding='same', name='convolution_4_1', kernel_initializer=initializer)(layer)
            layer = Convolution2D(128, 3, activation='relu', padding='same', name='convolution_4_2', kernel_initializer=initializer)(layer)
            layer = Convolution2D(128, 3, activation='relu', padding='same', name='convolution_4_3', kernel_initializer=initializer)(layer)
            layer = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_4')(layer)

            # Block 5
            layer = Convolution2D(256, 3, activation='relu', padding='same', name='convolution_5_1', kernel_initializer=initializer)(layer)
            layer = Convolution2D(256, 3, activation='relu', padding='same', name='convolution_5_2', kernel_initializer=initializer)(layer)
            layer = Convolution2D(256, 3, activation='relu', padding='same', name='convolution_5_3', kernel_initializer=initializer)(layer)
            layer = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_5')(layer)

            layer = Flatten(name='flatten_1')(layer)
            layer = Dense(32, activation='relu', name='dense_1', kernel_initializer=initializer)(layer)
            output_layer = Dense(2, activation='softmax', name='output', kernel_initializer=initializer)(layer)
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
            file_util.make_folders(network_path)
            model.save(network_path)
