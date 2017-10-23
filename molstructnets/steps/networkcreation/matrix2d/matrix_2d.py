from util import file_structure, file_util, logger, constants
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
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
            l = Dropout(0.3, name='dropout_input')(input_layer)
            l = Convolution2D(4, 4, activation='relu', name='convolution_1', kernel_initializer=initializer)(l)
            l = Dropout(0.75, name='dropout_convolution_1')(l)
            l = Convolution2D(8, 8, activation='relu', name='convolution_2', kernel_initializer=initializer)(l)
            l = Dropout(0.75, name='dropout_convolution_2')(l)
            l = Convolution2D(16, 16, activation='relu', name='convolution_3', kernel_initializer=initializer)(l)
            l = Dropout(0.75, name='dropout_convolution_3')(l)
            l = Convolution2D(32, 32, activation='relu', name='convolution_4', kernel_initializer=initializer)(l)
            l = Dropout(0.75, name='dropout_convolution_4')(l)
            l = Flatten(name='flatten_1')(l)
            l = Dense(128, activation='relu', name='dense_1', kernel_initializer=initializer)(l)
            l = Dropout(0.75, name='dropout_dense_1')(l)
            output_layer = Dense(2, activation='softmax', name='output', kernel_initializer=initializer)(l)
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            file_util.make_folders(network_path)
            model.save(network_path)
