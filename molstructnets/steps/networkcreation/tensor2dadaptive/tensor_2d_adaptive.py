from util import file_structure, file_util, logger, constants
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import initializers, optimizers


class Tensor2DAdaptive:

    @staticmethod
    def get_id():
        return 'tensor_2d_adaptive'

    @staticmethod
    def get_name():
        return '2D Tensor Adaptive'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'sparse', 'name': 'Sparse Input Features', 'type': bool, 'default': False,
                           'description': 'If the input features are sparse. In this case the network will start with'
                                          ' 2 features per position after the first convolution. Default: False'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        dimensions = global_parameters[constants.GlobalParameters.input_dimensions]
        if len(dimensions) != 3:
            raise ValueError('Preprocessed dimensions are not 2D')
        if dimensions[0] != dimensions[1]:
            raise ValueError('x and y dimensions do not match (' + str(dimensions[0]) + '!=' + str(dimensions[1]) + ')')

    @staticmethod
    def execute(global_parameters, local_parameters):
        network_path = file_structure.get_network_file(global_parameters)
        if file_util.file_exists(network_path):
            logger.log('Skipping step: ' + network_path + ' already exists')
        else:
            initializer = initializers.he_uniform()
            input_layer = Input(shape=global_parameters[constants.GlobalParameters.input_dimensions], name='input')
            layer = input_layer
            layer = Dropout(0.3, name='input_dropout')(layer)
            input_features = None
            if local_parameters['sparse']:
                input_features = 2
            iteration = 0
            while layer.shape[1] > 1:
                iteration += 1
                layer = add_block(layer, iteration, initializer, input_features=input_features)
                input_features = None
            layer = Flatten(name='features')(layer)
            layer = Dropout(0.75, name='dropout_1')(layer)
            layer = Dense(128, activation='relu', name='dense', kernel_initializer=initializer)(layer)
            layer = Dropout(0.75, name='dropout_2')(layer)
            output_layer = Dense(2, activation='softmax', name='output', kernel_initializer=initializer)(layer)
            model = Model(inputs=input_layer, outputs=output_layer)
            optimizer = optimizers.Adam(lr=0.0001)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
            file_util.make_folders(network_path)
            model.save(network_path)


def add_block(layer, iteration, initializer, input_features=None):
    if input_features is None:
        input_features = int(layer.shape[-1])
    layer = Convolution2D(input_features * 2, 3, activation='relu', padding='same', name='convolution_' + str(iteration),
                          kernel_initializer=initializer)(layer)
    return MaxPooling2D(2, padding='same', name='max_pool_' + str(iteration))(layer)
