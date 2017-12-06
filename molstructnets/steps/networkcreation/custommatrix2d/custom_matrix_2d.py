from util import file_structure, file_util, logger, constants
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import initializers


class CustomMatrix2D:

    @staticmethod
    def get_id():
        return 'custom_matrix_2d'

    @staticmethod
    def get_name():
        return 'Custom 2D Matrix'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'nr_blocks', 'name': 'Number blocks (default: 5)', 'type': int, 'default': 5,
                           'description': 'The number of blocks.'})
        parameters.append({'id': 'nr_convolutions', 'name': 'Number convolutions per block (default: 3)', 'type': int, 'default': 3,
                           'description': 'The number of convolutional layers per block.'})
        parameters.append({'id': 'base_convolution_output', 'name': 'Number base convolution outputs (default: 16)', 'type': int, 'default': 16,
                           'description': 'The number of outputs in the first convolutional layers. It will be multiplied by 2 for each additional block.'})
        parameters.append({'id': 'use_pooling', 'name': 'Use pooling (default: True)', 'type': bool, 'default': True,
                           'description': 'If pooling layers should be used at the end of each block.'})
        parameters.append({'id': 'dense_size', 'name': 'Number dense layer neurons (default: 128)', 'type': int, 'default': 128,
                           'description': 'Number of neurons in the hidden dense layer.'})
        parameters.append({'id': 'input_dropout', 'name': 'Input dropout (default: 0.3)', 'type': float, 'default': 0.3,
                           'description': 'Dropout on the input layer.'})
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
            layer = Dropout(local_parameters['input_dropout'], name='input_dropout')(layer)
            convolution_output_size = local_parameters['base_convolution_output']
            for i in range(local_parameters['nr_blocks']):
                CustomMatrix2D.add_block(layer, local_parameters['nr_convolutions'], convolution_output_size, local_parameters['use_pooling'], i, initializer)
                convolution_output_size *= 2
            layer = Flatten(name='flatten_1')(layer)
            layer = Dense(local_parameters['dense_size'], activation='relu', name='dense_1', kernel_initializer=initializer)(layer)
            output_layer = Dense(2, activation='softmax', name='output', kernel_initializer=initializer)(layer)
            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
            file_util.make_folders(network_path)
            model.save(network_path)

    @staticmethod
    def add_block(input_layer, nr_convolutions, convolution_output_size, use_pooling, block, initializer):
        layer = input_layer
        for i in range(nr_convolutions):
            layer = Convolution2D(convolution_output_size, 3, activation='relu', padding='same', name='convolution_' + str(block) + '_' + str(i), kernel_initializer=initializer)(layer)
        if use_pooling:
            layer = MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_' + str(block))(layer)
        return layer
