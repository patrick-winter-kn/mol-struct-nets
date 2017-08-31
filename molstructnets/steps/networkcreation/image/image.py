from util import file_structure, file_util, logger
from keras.models import Model
from keras.layers import Dropout, Conv2D, MaxPooling2D, Input, Dense, Flatten

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
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        dimensions = global_parameters['input_dimensions']
        if len(dimensions) != 3:
            raise ValueError('Preprocessed dimensions are not 2D')

    @staticmethod
    def execute(global_parameters, parameters):
        network_path = file_structure.get_network_file(global_parameters)
        if file_util.file_exists(network_path):
            logger.log('Skipping step: ' + network_path + ' already exists')
        else:
            img_input = Input(global_parameters['input_dimensions'], name='input')
            x = Dropout(0.3, name='input_drop')(img_input)
            x = Conv2D(64, 3, activation='relu', padding='same', name='conv_1')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_1')(x)
            x = Dropout(0.75, name='conv_1_drop')(x)
            x = Conv2D(128, 3, activation='relu', padding='same', name='conv_2')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_2')(x)
            x = Dropout(0.75, name='conv_2_drop')(x)
            x = Conv2D(256, 3, activation='relu', padding='same', name='conv_3')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_3')(x)
            x = Dropout(0.75, name='conv_3_drop')(x)
            x = Conv2D(512, 3, activation='relu', padding='same', name='conv_4')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_4')(x)
            x = Dropout(0.75, name='conv_4_drop')(x)
            x = Conv2D(512, 3, activation='relu', padding='same', name='conv_5')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_5')(x)
            x = Dropout(0.75, name='conv_5_drop')(x)
            x = Flatten(name='flatten')(x)
            x = Dense(32, activation='relu', name='dense_1')(x)
            x = Dropout(0.75, name='dense_1_drop')(x)
            out = Dense(2, activation='softmax', name='output')(x)
            model = Model(inputs=img_input, outputs=out)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.save(network_path)
