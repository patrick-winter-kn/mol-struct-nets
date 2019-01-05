import queue

import h5py
import numpy
from keras import models

from steps.preprocessing.shared.tensor2d import tensor_2d_array
from util import data_validation, file_structure, file_util, logger, progressbar, constants, hdf5_util, misc, \
    thread_pool


class LearnedFeatureGenerationTensor2D:

    @staticmethod
    def get_id():
        return 'learned_feature_generation_tensor_2d'

    @staticmethod
    def get_name():
        return 'Learned Features (Grid)'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'batch_size', 'name': 'Batch Size', 'type': int, 'default': 100, 'min': 1,
                           'description': 'Number of data points that will be processed together. A higher number leads'
                                          ' to faster processing but needs more memory. Default: 100'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_preprocessed_specs(global_parameters)
        data_validation.validate_network(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        file_name = 'learned_features.h5'
        return file_util.resolve_subpath(file_structure.get_result_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        model_path = file_structure.get_network_file(global_parameters)
        global_parameters[constants.GlobalParameters.feature_id] = 'learned_features'
        learned_features_path = LearnedFeatureGenerationTensor2D.get_result_file(global_parameters, local_parameters)
        model = models.load_model(model_path)
        feature_layer = model.get_layer('features')
        feature_dimensions = (int(numpy.prod(list(feature_layer.input.shape)[1:])),)
        if file_util.file_exists(learned_features_path):
            logger.log('Skipping step: ' + learned_features_path + ' already exists')
        else:
            feature_model = models.Model(inputs=model.input, outputs=feature_layer.output)
            array = tensor_2d_array.load_array(global_parameters)
            data_queue = queue.Queue(10)
            temp_learned_features_path = file_util.get_temporary_file_path('learned_features')
            learned_features_h5 = h5py.File(temp_learned_features_path, 'w')
            learned_features = hdf5_util.create_dataset(learned_features_h5, file_structure.Preprocessed.preprocessed,
                                                        (len(array),) + feature_dimensions, dtype='float16',
                                                        chunks=(1,) + feature_dimensions)
            logger.log('Generating features')
            chunks = misc.chunk_by_size(len(array), local_parameters['batch_size'])
            pool = thread_pool.ThreadPool(1)
            pool.submit(generate_data, array, chunks, data_queue)
            with progressbar.ProgressBar(len(array)) as progress:
                for chunk in chunks:
                    learned_features[chunk['start']:chunk['end']] = feature_model.predict(data_queue.get())[:]
                    progress.increment(chunk['size'])
            array.close()
            learned_features_h5.close()
            file_util.move_file(temp_learned_features_path, learned_features_path)
        global_parameters[constants.GlobalParameters.input_dimensions] = feature_dimensions
        global_parameters[constants.GlobalParameters.preprocessed_data] = learned_features_path
        global_parameters[constants.GlobalParameters.feature_files].append(learned_features_path)


def generate_data(array, chunks, data_queue):
    for chunk in chunks:
        data_queue.put(array[chunk['start']:chunk['end']])
