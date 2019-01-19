import queue

import h5py
import numpy

from steps.interpretation.shared.kerasviz import saliency_map
from steps.preprocessing.shared.tensor2d import tensor_2d_array
from util import data_validation, file_structure, file_util, progressbar, hdf5_util, logger, thread_pool


class CalculateSaliencyMaps2D:

    @staticmethod
    def get_id():
        return 'calculate_saliency_maps_2d'

    @staticmethod
    def get_name():
        return 'Calculate Saliency Maps'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'top_n', 'name': 'Top n', 'type': int, 'default': None, 'min': 1,
                           'description': 'A saliency map for the n highest scored molecules will be generated.'
                                          ' Default: All'})
        parameters.append({'id': 'actives', 'name': 'Active Class', 'type': bool, 'default': True,
                           'description': 'If true the saliency map will show the activation for the active class. If'
                                          ' false it will be for the inactive class. Default: True'})
        parameters.append({'id': 'correct_predictions', 'name': 'Only Correct Predictions', 'type': bool,
                           'default': False,
                           'description': 'If true only correct predictions will be considered. Default: False'})
        parameters.append({'id': 'partition', 'name': 'Partition', 'type': str, 'default': 'both',
                           'options': ['train', 'test', 'both'],
                           'description': 'Saliency maps will be generated for the specified partition. Options are:'
                                          ' train, test or both partitions. Default: both'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed_specs(global_parameters)
        data_validation.validate_network(global_parameters)
        data_validation.validate_prediction(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        saliency_map_path = file_structure.get_saliency_map_file(global_parameters)
        file_existed = file_util.file_exists(saliency_map_path)
        file_util.make_folders(saliency_map_path)
        saliency_map_h5 = h5py.File(saliency_map_path, 'a')
        if local_parameters['actives']:
            saliency_map_dataset_name = file_structure.SaliencyMap.saliency_map_active
        else:
            saliency_map_dataset_name = file_structure.SaliencyMap.saliency_map_inactive
        if saliency_map_dataset_name in saliency_map_h5.keys():
            logger.log('Skipping step: ' + saliency_map_dataset_name + ' in ' + saliency_map_path
                       + ' already exists')
            saliency_map_h5.close()
        else:
            saliency_map_h5.close()
            temp_saliency_map_path = file_util.get_temporary_file_path('saliency_map')
            if file_existed:
                file_util.copy_file(saliency_map_path, temp_saliency_map_path)
            else:
                file_util.remove_file(saliency_map_path)
            saliency_map_h5 = h5py.File(temp_saliency_map_path, 'a')
            target_h5 = h5py.File(file_structure.get_target_file(global_parameters), 'r')
            classes = target_h5[file_structure.Target.classes][:]
            target_h5.close()
            preprocessed = tensor_2d_array.load_array(global_parameters)
            partition_h5 = h5py.File(file_structure.get_partition_file(global_parameters), 'r')
            if local_parameters['partition'] == 'train':
                references = numpy.unique(partition_h5[file_structure.Partitions.train][:])
            elif local_parameters['partition'] == 'test':
                references = partition_h5[file_structure.Partitions.test][:]
            else:
                references = numpy.arange(0, len(preprocessed))
            partition_h5.close()
            if local_parameters['actives']:
                indices_data_set_name = file_structure.SaliencyMap.saliency_map_active_indices
                class_index = 0
            else:
                indices_data_set_name = file_structure.SaliencyMap.saliency_map_inactive_indices
                class_index = 1
            prediction_h5 = h5py.File(file_structure.get_prediction_file(global_parameters), 'r')
            predictions = prediction_h5[file_structure.Predictions.prediction][:, class_index]
            prediction_h5.close()
            if local_parameters['top_n'] is None:
                count = len(preprocessed)
                indices = references
            else:
                # Get class column ([:,0], sort it (.argsort()) and reverse the order ([::-1]))
                indices = predictions.argsort()[::-1]
                count = min(local_parameters['top_n'], len(indices))
            if saliency_map_dataset_name in saliency_map_h5.keys():
                saliency_map_ = saliency_map_h5[saliency_map_dataset_name]
            else:
                saliency_map_shape = list(preprocessed.shape)
                saliency_map_shape = saliency_map_shape[:-1]
                chunk_shape = tuple([1] + saliency_map_shape[1:])
                saliency_map_shape = tuple(saliency_map_shape)
                saliency_map_ = hdf5_util.create_dataset(saliency_map_h5, saliency_map_dataset_name, saliency_map_shape, chunks=chunk_shape,
                                                dtype='float32')
            saliency_map_indices_list = list()
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
                    if not numpy.max(saliency_map_[index][:]) > 0:
                        saliency_map_indices_list.append(index)
            if local_parameters['top_n'] is not None:
                saliency_map_indices_list = sorted(saliency_map_indices_list)
                hdf5_util.create_dataset_from_data(saliency_map_h5, indices_data_set_name, saliency_map_indices_list, dtype='uint32')
            data_queue = queue.Queue(10)
            saliency_map_calc = saliency_map.SaliencyMap(file_structure.get_network_file(global_parameters), class_index)
            with thread_pool.ThreadPool(1) as pool:
                pool.submit(generate_data, preprocessed, saliency_map_indices_list, data_queue)
                with progressbar.ProgressBar(len(saliency_map_indices_list)) as progress:
                    for i in range(len(saliency_map_indices_list)):
                        index = saliency_map_indices_list[i]
                        tensor = data_queue.get()
                        grads = saliency_map_calc.calculate(tensor)
                        saliency_map_[index] = grads[:] * predictions[index]
                        progress.increment()
            saliency_map_h5.close()
            file_util.move_file(temp_saliency_map_path, saliency_map_path)


def generate_data(preprocessed, saliency_map_indices_list, queue):
    for i in range(len(saliency_map_indices_list)):
        queue.put(preprocessed[saliency_map_indices_list[i]])
