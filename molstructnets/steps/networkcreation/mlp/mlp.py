from keras import initializers, optimizers
from keras.layers import Input
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Model

from util import file_structure, file_util, logger, constants


class MLP:

    @staticmethod
    def get_id():
        return 'mlp'

    @staticmethod
    def get_name():
        return 'Multi Layer Perceptron'

    @staticmethod
    def get_parameters():
        parameters = list()
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        pass

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
            if len(global_parameters[constants.GlobalParameters.input_dimensions]) > 1:
                layer = Flatten(name='flatten_1')(layer)
            layer = Dense(128, activation='relu', name='dense', kernel_initializer=initializer)(layer)
            output_layer = Dense(2, activation='softmax', name='output', kernel_initializer=initializer)(layer)
            model = Model(inputs=input_layer, outputs=output_layer)
            optimizer = optimizers.Adam(lr=0.0001)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
            file_util.make_folders(network_path)
            model.save(network_path)
