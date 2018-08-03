import h5py
import numpy
from rdkit import Chem

from steps.interpretation.extractcamsubstructures2djit import substructure_set
from steps.preprocessing.shared.tensor2d import tensor_2d_jit_array
from util import data_validation, file_structure, file_util, progressbar, logger, misc, hdf5_util, buffered_queue


class ExtractCamSubstructures2DJit:

    @staticmethod
    def get_id():
        return 'extract_cam_substructures_2d_jit'

    @staticmethod
    def get_name():
        return 'Extract CAM Substructures 2D JIT'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'threshold', 'name': 'Threshold', 'type': float, 'default': 0.25, 'min': 0.0,
                           'max': 1.0, 'description': 'The threshold used to decide which parts of the CAM'
                                                      ' are interpreted as part of the substructure. Default: 0.25'})
        parameters.append({'id': 'partition', 'name': 'Partition', 'type': str, 'default': 'both',
                           'options': ['train', 'test', 'both'],
                           'description': 'The partition that the substructures will be extracted from. Options are:'
                                          ' train, test or both partitions. Default: both'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_preprocessed_jit(global_parameters)
        data_validation.validate_cam(global_parameters)
        if local_parameters['partition'] != 'both':
            data_validation.validate_partition(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        cam_substructures_path = file_util.resolve_subpath(
            file_structure.get_interpretation_folder(global_parameters), 'cam_substructures.h5')
        if file_util.file_exists(cam_substructures_path):
            logger.log('Skipping step: ' + cam_substructures_path + ' already exists')
        else:
            cam_h5 = h5py.File(file_structure.get_cam_file(global_parameters), 'r')
            array = tensor_2d_jit_array.load_array(global_parameters)
            temp_cam_substructures_path = file_util.get_temporary_file_path(
                'cam_substructures')
            cam_substructures_h5 = h5py.File(temp_cam_substructures_path, 'w')
            if file_structure.Cam.cam_active in cam_h5.keys():
                ExtractCamSubstructures2DJit.extract_cam_substructures(
                    cam_h5, array, global_parameters, local_parameters, cam_substructures_h5, True)
            if file_structure.Cam.cam_inactive in cam_h5.keys():
                ExtractCamSubstructures2DJit.extract_cam_substructures(
                    cam_h5, array, global_parameters, local_parameters, cam_substructures_h5, False)
            cam_substructures_h5.close()
            file_util.move_file(temp_cam_substructures_path, cam_substructures_path)
            cam_h5.close()

    @staticmethod
    def extract_cam_substructures(cam_h5, array, global_parameters, local_parameters, cam_substructures_h5, active):
        if active:
            log_message = 'Extracting active CAM substructures'
            cam_dataset_name = file_structure.Cam.cam_active
            cam_indices_dataset_name = file_structure.Cam.cam_active_indices
            substructures_dataset_name = 'active_substructures'
            substructures_occurrences_dataset_name = 'active_substructures_occurrences'
            substructures_value_dataset_name = 'active_substructures_value'
            substructures_number_heavy_atoms_dataset_name = 'active_substructures_number_heavy_atoms'
            substructures_score_dataset_name = 'active_substructures_score'
        else:
            log_message = 'Extracting inactive CAM substructures'
            cam_dataset_name = file_structure.Cam.cam_inactive
            cam_indices_dataset_name = file_structure.Cam.cam_inactive_indices
            substructures_dataset_name = 'inactive_substructures'
            substructures_occurrences_dataset_name = 'inactive_substructures_occurrences'
            substructures_value_dataset_name = 'inactive_substructures_value'
            substructures_number_heavy_atoms_dataset_name = 'inactive_substructures_number_heavy_atoms'
            substructures_score_dataset_name = 'inactive_substructures_score'
        cam = cam_h5[cam_dataset_name]
        if cam_indices_dataset_name in cam_h5.keys():
            indices = cam_h5[cam_indices_dataset_name]
        else:
            indices = range(len(cam))
        if local_parameters['partition'] != 'both':
            partition_h5 = h5py.File(file_structure.get_partition_file(global_parameters), 'r')
            if local_parameters['partition'] == 'train':
                train = partition_h5[file_structure.Partitions.train]
                indices = list(set(indices) & set(numpy.array(train).flatten()))
            elif local_parameters['partition'] == 'test':
                test = partition_h5[file_structure.Partitions.test]
                indices = list(set(indices) & set(numpy.array(test).flatten()))
            partition_h5.close()
        logger.log(log_message, logger.LogLevel.INFO)
        substructures = substructure_set.SubstructureSet()
        with progressbar.ProgressBar(len(indices)) as progress:
            ExtractCamSubstructures2DJit.extract(cam, indices, array, substructures, local_parameters['threshold'],
                                                 progress)
        substructures_dict = substructures.get_dict()
        substructures = list(substructures_dict.keys())
        max_length = 0
        for smiles_string in substructures:
            max_length = max(max_length, len(smiles_string))
        dtype = 'S' + str(max(max_length, 1))
        substructures_dataset = hdf5_util.create_dataset(cam_substructures_h5, substructures_dataset_name,
                                                         (len(substructures),), dtype=dtype)
        substructures_occurrences_dataset = hdf5_util.create_dataset(cam_substructures_h5,
                                                                     substructures_occurrences_dataset_name,
                                                                     (len(substructures),), dtype='I')
        substructures_value_dataset = hdf5_util.create_dataset(cam_substructures_h5,
                                                               substructures_value_dataset_name, (len(substructures),))
        substructures_number_heavy_atoms_dataset = hdf5_util.create_dataset(cam_substructures_h5,
                                                                            substructures_number_heavy_atoms_dataset_name,
                                                                            (len(substructures),), dtype='I')
        substructures_score_dataset = hdf5_util.create_dataset(cam_substructures_h5,
                                                               substructures_score_dataset_name, (len(substructures),))
        occurences = numpy.zeros(len(substructures))
        values = numpy.zeros(len(substructures))
        number_heavy_atoms = numpy.zeros(len(substructures))
        for i in range(len(substructures)):
            substructure = substructures_dict[substructures[i]]
            occurences[i] = substructure.get_occurrences()
            values[i] = substructure.get_mean_value()
            number_heavy_atoms[i] = substructure.get_number_heavy_atoms()
        misc.normalize(occurences)
        misc.normalize(values)
        misc.normalize(number_heavy_atoms)
        scores = numpy.zeros(len(substructures))
        for i in range(len(substructures)):
            scores[i] = occurences[i] * values[i] * number_heavy_atoms[i]
        misc.normalize(scores)
        sorted_indices = scores.argsort()[::-1]
        i = 0
        for j in sorted_indices:
            substructures_dataset[i] = substructures[j].encode()
            substructure = substructures_dict[substructures[j]]
            substructures_occurrences_dataset[i] = substructure.get_occurrences()
            substructures_value_dataset[i] = substructure.get_mean_value()
            substructures_number_heavy_atoms_dataset[i] = substructure.get_number_heavy_atoms()
            substructures_score_dataset[i] = scores[j]
            i += 1

    @staticmethod
    def extract(cam, indices, array, substructures, threshold, progress):
        location_queue = buffered_queue.BufferedQueue(1, 10000)
        array.calc_atom_locations(0, len(array), location_queue)
        for i in range(len(array)):
            index, locations = location_queue.get()
            if index in indices:
                smiles_string = array.smiles(index).decode('utf-8')
                atom_indices, values = ExtractCamSubstructures2DJit.pick_atoms(cam[index], threshold, locations)
                if len(atom_indices) > 0:
                    molecule = Chem.MolFromSmiles(smiles_string)
                    ExtractCamSubstructures2DJit.add_substructures(molecule, atom_indices, values, substructures)
                progress.increment()

    @staticmethod
    def pick_atoms(cam, threshold, atom_locations):
        picked_indices = list()
        values = list()
        for i in range(len(atom_locations)):
            if atom_locations[i][0] < 0:
                break
            location = tuple(atom_locations[i])
            value = cam[location]
            if value >= threshold:
                picked_indices.append(i)
                values.append(value)
        return picked_indices, values

    @staticmethod
    def add_substructures(molecule, atom_indices, values, substructure):
        while len(atom_indices) > 0:
            indices = {atom_indices[0]}
            new_indices = {atom_indices[0]}
            value_sum = 0
            while len(new_indices) > 0:
                next_new_indices = set()
                for index in new_indices:
                    for neighbor in molecule.GetAtoms()[index].GetNeighbors():
                        neighbor_index = neighbor.GetIdx()
                        if neighbor_index in atom_indices:
                            if neighbor_index not in indices:
                                indices.add(neighbor_index)
                                next_new_indices.add(neighbor_index)
                new_indices = next_new_indices
            for index in indices:
                list_index = atom_indices.index(index)
                value_sum += values[list_index]
                del atom_indices[list_index]
                del values[list_index]
            value_mean = value_sum / len(indices)
            smiles = Chem.MolFragmentToSmiles(molecule, indices)
            substructure.add_substructure(smiles, value_mean)
