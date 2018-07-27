import math

import h5py
import numpy
from keras import models

from util import data_validation, file_structure, file_util, logger, progressbar, constants, hdf5_util


class LearnedFeatureGeneration:

    @staticmethod
    def get_id():
        return 'learned_feature_generation'

    @staticmethod
    def get_name():
        return 'Learned Feature Generation'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'batch_size', 'name': 'Batch Size', 'type': int, 'default': 50, 'min': 1,
                           'description': 'Number of data points that will be processed together. A higher number leads'
                                          ' to faster processing but needs more memory. Default: 50'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_preprocessed(global_parameters)
        data_validation.validate_network(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        file_name = 'learned_features.h5'
        return file_util.resolve_subpath(file_structure.get_result_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        learned_features_path = LearnedFeatureGeneration.get_result_file(global_parameters, local_parameters)
        model_path = file_structure.get_network_file(global_parameters)
        model = models.load_model(model_path)
        feature_layer = model.get_layer('features')
        feature_dimensions = (int(numpy.prod(list(feature_layer.input.shape)[1:])),)
        if file_util.file_exists(learned_features_path):
            logger.log('Skipping step: ' + learned_features_path + ' already exists')
        else:
            feature_model = models.Model(inputs=model.input, outputs=feature_layer.output)
            preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data], 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            temp_learned_features_path = file_util.get_temporary_file_path('learned_features')
            learned_features_h5 = h5py.File(temp_learned_features_path, 'w')
            learned_features = hdf5_util.create_dataset(learned_features_h5, file_structure.Preprocessed.preprocessed,
                                                        (len(preprocessed),) + feature_dimensions,
                                                        chunks=(1,) + feature_dimensions)
            logger.log('Generating features')
            with progressbar.ProgressBar(len(preprocessed)) as progress:
                for i in range(int(math.ceil(len(preprocessed) / local_parameters['batch_size']))):
                    start = i * local_parameters['batch_size']
                    end = min(len(preprocessed), (i + 1) * local_parameters['batch_size'])
                    results = feature_model.predict(preprocessed[start:end])
                    learned_features[start:end] = results[:]
                    progress.increment(end - start)
            preprocessed_h5.close()
            learned_features_h5.close()
            file_util.move_file(temp_learned_features_path, learned_features_path)
        global_parameters[constants.GlobalParameters.input_dimensions] = feature_dimensions
        global_parameters[constants.GlobalParameters.preprocessed_data] = learned_features_path
        if constants.GlobalParameters.preprocessed_training_data in global_parameters:
            del global_parameters[constants.GlobalParameters.preprocessed_training_data]
