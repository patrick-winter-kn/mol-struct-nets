import queue

import h5py
import numpy

from steps.interpretation.shared.kerasviz import cam
from steps.preprocessing.shared.tensor2d import tensor_2d_array
from util import data_validation, file_structure, file_util, progressbar, hdf5_util, logger, thread_pool


class CalculateCams2D:

    @staticmethod
    def get_id():
        return 'calculate_cams_2d'

    @staticmethod
    def get_name():
        return 'Calculate CAMs 2D'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'top_n', 'name': 'Top n', 'type': int, 'default': None, 'min': 1,
                           'description': 'A CAM for the n highest scored molecules will be generated.'
                                          ' Default: All'})
        parameters.append({'id': 'actives', 'name': 'Active Class', 'type': bool, 'default': True,
                           'description': 'If true the CAM will show the activation for the active class. If'
                                          ' false it will be for the inactive class. Default: True'})
        parameters.append({'id': 'correct_predictions', 'name': 'Only Correct Predictions', 'type': bool,
                           'default': False,
                           'description': 'If true only correct predictions will be considered. Default: False'})
        parameters.append({'id': 'partition', 'name': 'Partition', 'type': str, 'default': 'both',
                           'options': ['train', 'test', 'both'],
                           'description': 'CAMs will be generated for the specified partition. Options are:'
                                          ' train, test or both partitions. Default: both'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed_jit(global_parameters)
        data_validation.validate_network(global_parameters)
        data_validation.validate_prediction(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        cam_path = file_structure.get_cam_file(global_parameters)
        file_existed = file_util.file_exists(cam_path)
        file_util.make_folders(cam_path)
        cam_h5 = h5py.File(cam_path, 'a')
        if local_parameters['actives']:
            cam_dataset_name = file_structure.Cam.cam_active
        else:
            cam_dataset_name = file_structure.Cam.cam_inactive
        if cam_dataset_name in cam_h5.keys():
            logger.log('Skipping step: ' + cam_dataset_name + ' in ' + cam_path
                       + ' already exists')
            cam_h5.close()
        else:
            cam_h5.close()
            temp_cam_path = file_util.get_temporary_file_path('cam')
            if file_existed:
                file_util.copy_file(cam_path, temp_cam_path)
            else:
                file_util.remove_file(cam_path)
            cam_h5 = h5py.File(temp_cam_path, 'a')
            target_h5 = h5py.File(file_structure.get_target_file(global_parameters), 'r')
            classes = target_h5[file_structure.Target.classes][:]
            target_h5.close()
            preprocessed = tensor_2d_array.load_array(global_parameters)
            partition_h5 = h5py.File(file_structure.get_partition_file(global_parameters), 'r')
            if local_parameters['partition'] == 'train':
                references = partition_h5[file_structure.Partitions.train][:]
            elif local_parameters['partition'] == 'test':
                references = partition_h5[file_structure.Partitions.test][:]
            else:
                references = numpy.arange(0, len(preprocessed))
            partition_h5.close()
            if local_parameters['actives']:
                indices_data_set_name = file_structure.Cam.cam_active_indices
                class_index = 0
            else:
                indices_data_set_name = file_structure.Cam.cam_inactive_indices
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
            if cam_dataset_name in cam_h5.keys():
                cam_ = cam_h5[cam_dataset_name]
            else:
                cam_shape = list(preprocessed.shape)
                cam_shape = cam_shape[:-1]
                chunk_shape = tuple([1] + cam_shape[1:])
                cam_shape = tuple(cam_shape)
                cam_ = hdf5_util.create_dataset(cam_h5, cam_dataset_name, cam_shape, chunks=chunk_shape,
                                                dtype='float32')
            cam_indices_list = list()
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
                    if not numpy.max(cam_[index][:]) > 0:
                        cam_indices_list.append(index)
            if local_parameters['top_n'] is not None:
                cam_indices_list = sorted(cam_indices_list)
                hdf5_util.create_dataset_from_data(cam_h5, indices_data_set_name, cam_indices_list, dtype='uint32')
            data_queue = queue.Queue(10)
            cam_calc = cam.CAM(file_structure.get_network_file(global_parameters), class_index)
            with thread_pool.ThreadPool(1) as pool:
                pool.submit(generate_data, preprocessed, cam_indices_list, data_queue)
                with progressbar.ProgressBar(len(cam_indices_list)) as progress:
                    for i in range(len(cam_indices_list)):
                        index = cam_indices_list[i]
                        tensor = data_queue.get()
                        grads = cam_calc.calculate(tensor)
                        cam_[index] = grads[:] * predictions[index]
                        progress.increment()
            cam_h5.close()
            file_util.move_file(temp_cam_path, cam_path)


def generate_data(preprocessed, cam_indices_list, queue):
    for i in range(len(cam_indices_list)):
        queue.put(preprocessed[cam_indices_list[i]])
