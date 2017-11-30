import h5py
from keras import backend
from keras import models, activations
from vis.utils import utils
import numpy

from steps.interpretation.shared.kerasviz import attention_map
from util import data_validation, file_structure, file_util, progressbar, misc, constants, hdf5_util, logger


class CalculateAttentionMaps:

    iterations_per_clear = 20

    @staticmethod
    def get_id():
        return 'calculate_attention_maps'

    @staticmethod
    def get_name():
        return 'Calculate Attention Maps'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'top_n', 'name': 'Top n (default: all)', 'type': int, 'default': None,
                           'description': 'An attention map for the n highest scored molecules will be generated.'})
        parameters.append({'id': 'actives', 'name': 'Active class (otherwise inactive, default: True)', 'type': bool,
                           'default': True,
                           'description': 'If true the attention map will show the attention for the active class. If'
                                          ' false it will be for the inactive class.'})
        parameters.append({'id': 'correct_predictions', 'name': 'Only correct predictions (default: False)',
                           'type': bool, 'default': False,
                           'description': 'If true only correct predictions will be considered.'})
        parameters.append({'id': 'partition', 'name': 'Partition (options: train, test or both, default: both)',
                           'type': str, 'default': 'both',
                           'description': 'Attention maps will be generated for the specified partition. By default'
                                          ' both the train and test partition will be used.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed(global_parameters)
        data_validation.validate_network(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        attention_map_path = file_structure.get_attentionmap_file(global_parameters)
        file_existed = file_util.file_exists(attention_map_path)
        file_util.make_folders(attention_map_path)
        attention_map_h5 = h5py.File(attention_map_path, 'a')
        if local_parameters['actives']:
            attention_map_dataset_name = file_structure.AttentionMap.attention_map_active
        else:
            attention_map_dataset_name = file_structure.AttentionMap.attention_map_inactive
        if attention_map_dataset_name in attention_map_h5.keys():
            logger.log('Skipping step: ' + attention_map_dataset_name + ' in ' + attention_map_path
                       + ' already exists')
            attention_map_h5.close()
        else:
            attention_map_h5.close()
            temp_attention_map_path = file_util.get_temporary_file_path('attention_map')
            if file_existed:
                file_util.copy_file(attention_map_path, temp_attention_map_path)
            else:
                file_util.remove_file(attention_map_path)
            attention_map_h5 = h5py.File(temp_attention_map_path, 'a')
            modified_model_path = file_util.get_temporary_file_path('modified_model')
            model = models.load_model(file_structure.get_network_file(global_parameters))
            out_layer_index = len(model.layers)-1
            model.layers[out_layer_index].activation = activations.linear
            model = utils.apply_modifications(model)
            model.save(modified_model_path)
            target_h5 = h5py.File(file_structure.get_target_file(global_parameters), 'r')
            classes = target_h5[file_structure.Target.classes]
            prediction_h5 = h5py.File(file_structure.get_prediction_file(global_parameters), 'r')
            predictions = prediction_h5[file_structure.Predictions.prediction]
            partition_h5 = h5py.File(file_structure.get_partition_file(global_parameters), 'r')
            preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data], 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            if local_parameters['partition'] == 'train':
                references = partition_h5[file_structure.Partitions.train]
            elif local_parameters['partition'] == 'test':
                references = partition_h5[file_structure.Partitions.test]
            else:
                references = numpy.arange(0, len(preprocessed))
            # Speedup lookup by copying into memory
            references = misc.copy_into_memory(references)
            if local_parameters['top_n'] is None:
                count = len(preprocessed)
                indices = references
                attention_map_indices_list = None
                attention_map_indices = None
            else:
                # We copy the needed data into memory to speed up sorting
                # Get first column ([:,0], sort it (.argsort()) and reverse the order ([::-1]))
                indices = misc.copy_into_memory(predictions)[:, 0].argsort()[::-1]
                count = min(local_parameters['top_n'], len(indices))
                if local_parameters['actives']:
                    indices_data_set_name = file_structure.AttentionMap.attention_map_active_indices
                else:
                    indices_data_set_name = file_structure.AttentionMap.attention_map_inactive_indices
                attention_map_indices_list = list()
                attention_map_indices = hdf5_util.create_dataset(attention_map_h5, indices_data_set_name, (count,),
                                                                 dtype='I')
            if local_parameters['actives']:
                class_index = 0
            else:
                class_index = 1
            if attention_map_dataset_name in attention_map_h5.keys():
                attention_map_ = attention_map_h5[attention_map_dataset_name]
            else:
                attention_map_shape = list(preprocessed.shape)
                attention_map_shape = tuple(attention_map_shape[:-1])
                attention_map_ = hdf5_util.create_dataset(attention_map_h5, attention_map_dataset_name,
                                                          attention_map_shape)
            with progressbar.ProgressBar(count) as progress:
                j = 0
                for i in range(count):
                    index = -1
                    while index is not None and index not in references:
                        if j >= len(indices):
                            index = None
                        else:
                            index = indices[j]
                            if local_parameters['correct_predictions']:
                                if classes[index][class_index] != 1:
                                    index = -1
                        j += 1
                    if index is not None:
                        if not numpy.max(attention_map_[index]) > 0:
                            matrix = preprocessed[index]
                            grads = attention_map.calculate_saliency(model, out_layer_index,
                                                                     filter_indices=[class_index],
                                                                     seed_input=matrix)
                            attention_map_[index] = grads[:]
                            if attention_map_indices_list is not None:
                                attention_map_indices_list.append(index)
                            if i % CalculateAttentionMaps.iterations_per_clear == 0:
                                backend.clear_session()
                                model = models.load_model(modified_model_path)
                    progress.increment()
            if attention_map_indices_list is not None:
                attention_map_indices_list = sorted(attention_map_indices_list)
                attention_map_indices[:] = attention_map_indices_list[:]
            attention_map_h5.close()
            target_h5.close()
            prediction_h5.close()
            partition_h5.close()
            preprocessed_h5.close()
            file_util.move_file(temp_attention_map_path, attention_map_path)
