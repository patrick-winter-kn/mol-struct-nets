from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

from util import file_structure, file_util, logger, constants


class Vgg19:

    @staticmethod
    def get_id():
        return 'vgg19'

    @staticmethod
    def get_name():
        return 'VGG19'

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
            img_input = Input(shape=global_parameters[constants.GlobalParameters.input_dimensions])
            x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
            x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
            x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
            x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
            x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
            x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
            x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
            x = Flatten(name='flatten')(x)
            x = Dense(32, activation='relu', name='fc1')(x)
            x = Dense(2, activation='softmax', name='predictions')(x)
            model = Model(inputs=img_input, outputs=x)
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            file_util.make_folders(network_path)
            model.save(network_path)
