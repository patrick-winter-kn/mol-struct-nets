from util import file_structure, file_util, logger, constants
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution1D
from keras import initializers


class TensorSmiles:

    @staticmethod
    def get_id():
        return 'tensor_smiles'

    @staticmethod
    def get_name():
        return 'SMILES Tensor'

    @staticmethod
    def get_parameters():
        parameters = list()
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        dimensions = global_parameters[constants.GlobalParameters.input_dimensions]
        if len(dimensions) != 2:
            raise ValueError('Preprocessed dimensions are not 1D')

    @staticmethod
    def execute(global_parameters, local_parameters):
        network_path = file_structure.get_network_file(global_parameters)
        if file_util.file_exists(network_path):
            logger.log('Skipping step: ' + network_path + ' already exists')
        else:
            initializer = initializers.he_uniform()
            input_layer = Input(shape=global_parameters[constants.GlobalParameters.input_dimensions], name='input')
            layer = Dropout(0.3, name='dropout_input')(input_layer)
            layer = Convolution1D(4, 4, activation='relu', name='convolution_1', kernel_initializer=initializer)(layer)
            layer = Dropout(0.75, name='dropout_convolution_1')(layer)
            layer = Convolution1D(8, 8, activation='relu', name='convolution_2', kernel_initializer=initializer)(layer)
            layer = Dropout(0.75, name='dropout_convolution_2')(layer)
            layer = Convolution1D(16, 16, activation='relu', name='convolution_3',
                                  kernel_initializer=initializer)(layer)
            layer = Dropout(0.75, name='dropout_convolution_3')(layer)
            layer = Convolution1D(32, 32, activation='relu', name='convolution_4',
                                  kernel_initializer=initializer)(layer)
            layer = Dropout(0.75, name='dropout_convolution_4')(layer)
            layer = Flatten(name='features')(layer)
            layer = Dense(128, activation='relu', name='dense_1', kernel_initializer=initializer)(layer)
            layer = Dropout(0.75, name='dropout_dense_1')(layer)
            output_layer = Dense(2, activation='softmax', name='output', kernel_initializer=initializer)(layer)
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            file_util.make_folders(network_path)
            model.save(network_path)
