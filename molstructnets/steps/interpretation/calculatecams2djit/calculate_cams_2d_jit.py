import h5py
from keras import backend
from keras import models, activations
from vis.utils import utils
import numpy

from steps.interpretation.shared.kerasviz import cam
from util import data_validation, file_structure, file_util, progressbar, hdf5_util, logger
from steps.preprocessing.shared.tensor2d import tensor_2d_jit_array


class CalculateCams2DJit:

    iterations_per_clear = 20

    @staticmethod
    def get_id():
        return 'calculate_cams_2d_jit'

    @staticmethod
    def get_name():
        return 'Calculate CAMs 2D JIT'

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
            modified_model_path = file_util.get_temporary_file_path('modified_model')
            model = models.load_model(file_structure.get_network_file(global_parameters))
            out_layer_index = len(model.layers)-1
            model.layers[out_layer_index].activation = activations.linear
            model = utils.apply_modifications(model)
            model.save(modified_model_path)
            target_h5 = h5py.File(file_structure.get_target_file(global_parameters), 'r')
            classes = target_h5[file_structure.Target.classes][:]
            prediction_h5 = h5py.File(file_structure.get_prediction_file(global_parameters), 'r')
            predictions = prediction_h5[file_structure.Predictions.prediction][:]
            partition_h5 = h5py.File(file_structure.get_partition_file(global_parameters), 'r')
            preprocessed = tensor_2d_jit_array.load_array(global_parameters)
            if local_parameters['partition'] == 'train':
                references = partition_h5[file_structure.Partitions.train][:]
            elif local_parameters['partition'] == 'test':
                references = partition_h5[file_structure.Partitions.test][:]
            else:
                references = numpy.arange(0, len(preprocessed))
            cam_indices_list = list()
            if local_parameters['top_n'] is None:
                count = len(preprocessed)
                indices = references
            else:
                # Get first column ([:,0], sort it (.argsort()) and reverse the order ([::-1]))
                indices = predictions[:, 0].argsort()[::-1]
                count = min(local_parameters['top_n'], len(indices))
                if local_parameters['actives']:
                    indices_data_set_name = file_structure.Cam.cam_active_indices
                else:
                    indices_data_set_name = file_structure.Cam.cam_inactive_indices
            if local_parameters['actives']:
                class_index = 0
            else:
                class_index = 1
            if cam_dataset_name in cam_h5.keys():
                cam_ = cam_h5[cam_dataset_name]
            else:
                cam_shape = list(preprocessed.shape)
                cam_shape = tuple(cam_shape[:-1])
                cam_ = hdf5_util.create_dataset(cam_h5, cam_dataset_name,
                                                          cam_shape)
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
                cam_indices = hdf5_util.create_dataset(cam_h5, indices_data_set_name,
                                                                 (len(cam_indices_list),), dtype='I')
                cam_indices_list = sorted(cam_indices_list)
                cam_indices[:] = cam_indices_list[:]
            with progressbar.ProgressBar(len(cam_indices_list)) as progress:
                for i in range(len(cam_indices_list)):
                    index = cam_indices_list[i]
                    tensor = preprocessed[index]
                    grads = cam.calculate_saliency(model, out_layer_index,
                                                   filter_indices=[class_index],
                                                   seed_input=tensor)
                    cam_[index] = grads[:]
                    if i % CalculateCams2DJit.iterations_per_clear == 0:
                        backend.clear_session()
                        model = models.load_model(modified_model_path)
                    progress.increment()
            cam_h5.close()
            target_h5.close()
            prediction_h5.close()
            partition_h5.close()
            file_util.move_file(temp_cam_path, cam_path)
