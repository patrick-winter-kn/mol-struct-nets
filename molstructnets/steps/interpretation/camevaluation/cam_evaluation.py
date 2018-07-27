import warnings

import h5py
import numpy

from util import data_validation, file_structure, file_util, logger, hdf5_util, smiles_analyzer, progressbar, constants


class CamEvaluation:

    @staticmethod
    def get_id():
        return 'cam_evaluation'

    @staticmethod
    def get_name():
        return 'CAM Evaluation'

    @staticmethod
    def get_parameters():
        parameters = list()
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_preprocessed(global_parameters)
        data_validation.validate_cam(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        cam_evaluation_active_path = file_util.resolve_subpath(
            file_structure.get_interpretation_folder(global_parameters), 'cam_evaluation_active.h5')
        cam_evaluation_inactive_path = file_util.resolve_subpath(
            file_structure.get_interpretation_folder(global_parameters), 'cam_evaluation_inactive.h5')
        cam_h5 = h5py.File(file_structure.get_cam_file(global_parameters), 'r')
        preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data])
        atom_locations = None
        if file_structure.Preprocessed.atom_locations in preprocessed_h5.keys():
            atom_locations = preprocessed_h5[file_structure.Preprocessed.atom_locations]
        substructure_atoms = cam_h5[file_structure.Cam.substructure_atoms]
        data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
        smiles = data_h5[file_structure.DataSet.smiles]
        if file_structure.Cam.cam_active in cam_h5.keys():
            if file_util.file_exists(cam_evaluation_active_path):
                logger.log('Skipping: ' + cam_evaluation_active_path + ' already exists')
            else:
                indices = None
                if file_structure.Cam.cam_active_indices in cam_h5.keys():
                    indices = cam_h5[file_structure.Cam.cam_active_indices]
                temp_cam_evaluation_active_path = file_util.get_temporary_file_path('cam_evaluation_active')
                cam_active = cam_h5[file_structure.Cam.cam_active]
                CamEvaluation.calculate_cam_evaluation(cam_active, substructure_atoms, smiles,
                                                       temp_cam_evaluation_active_path, indices,
                                                       atom_locations)
                file_util.move_file(temp_cam_evaluation_active_path, cam_evaluation_active_path)
        if file_structure.Cam.cam_inactive in cam_h5.keys():
            if file_util.file_exists(cam_evaluation_inactive_path):
                logger.log('Skipping: ' + cam_evaluation_inactive_path + ' already exists')
            else:
                indices = None
                if file_structure.Cam.cam_inactive_indices in cam_h5.keys():
                    indices = cam_h5[file_structure.Cam.cam_inactive_indices]
                temp_cam_evaluation_inactive_path = file_util.get_temporary_file_path(
                    'cam_evaluation_inactive')
                cam_inactive = cam_h5[file_structure.Cam.cam_inactive]
                CamEvaluation.calculate_cam_evaluation(cam_inactive, substructure_atoms, smiles,
                                                       temp_cam_evaluation_inactive_path, indices,
                                                       atom_locations)
                file_util.move_file(temp_cam_evaluation_inactive_path, cam_evaluation_inactive_path)
        data_h5.close()
        cam_h5.close()
        preprocessed_h5.close()

    @staticmethod
    def calculate_cam_evaluation(cam, substructure_atoms, smiles, cam_evaluation_path, indices,
                                 atom_locations=None):
        if indices is None:
            indices = range(len(cam))
        cam_evaluation_h5 = h5py.File(cam_evaluation_path, 'w')
        characters = hdf5_util.create_dataset(cam_evaluation_h5, 'characters', (len(indices),), dtype='I')
        mean = hdf5_util.create_dataset(cam_evaluation_h5, 'mean', (len(indices),))
        std_deviation = hdf5_util.create_dataset(cam_evaluation_h5, 'std_deviation', (len(indices),))
        substructure_characters = hdf5_util.create_dataset(cam_evaluation_h5, 'substructure_characters',
                                                           (len(indices),), dtype='I')
        substructure_mean = hdf5_util.create_dataset(cam_evaluation_h5, 'substructure_mean', (len(indices),))
        substructure_std_deviation = hdf5_util.create_dataset(cam_evaluation_h5, 'substructure_std_deviation',
                                                              (len(indices),))
        distance = hdf5_util.create_dataset(cam_evaluation_h5, 'distance', (len(indices),))
        index = 0
        with progressbar.ProgressBar(len(indices)) as progress:
            for i in indices:
                locations = None
                if atom_locations is not None:
                    locations = atom_locations[i]
                characters[index], mean[index], std_deviation[index], substructure_characters[index], \
                substructure_mean[index], substructure_std_deviation[index] = \
                    CamEvaluation.calculate_single_cam_evaluation(smiles[i].decode('utf-8'),
                                                                  substructure_atoms[i],
                                                                  cam[i], locations)
                distance[index] = CamEvaluation.calculate_single_distance(smiles[i].decode('utf-8'),
                                                                          substructure_atoms[i], cam[i],
                                                                          locations)
                index += 1
                progress.increment()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            overall_mean = numpy.mean(mean)
            overall_substructure_mean = numpy.mean(substructure_mean)
            overall_mean_std_deviation = numpy.std(mean)
            overall_substructure_mean_std_deviation = numpy.std(substructure_mean)
        cam_evaluation_h5.close()
        hdf5_util.set_property(cam_evaluation_path, 'overall_mean', overall_mean)
        hdf5_util.set_property(cam_evaluation_path, 'overall_substructure_mean', overall_substructure_mean)
        hdf5_util.set_property(cam_evaluation_path, 'overall_mean_std_deviation', overall_mean_std_deviation)
        hdf5_util.set_property(cam_evaluation_path, 'overall_substructure_mean_std_deviation',
                               overall_substructure_mean_std_deviation)

    @staticmethod
    def calculate_single_cam_evaluation(smiles_string, substructure_atoms, cam, atom_locations=None):
        substructure_values = list()
        not_substructure_values = list()
        if atom_locations is None:
            positions = smiles_analyzer.atom_positions(smiles_string)
            character_positions = set()
            for position in positions:
                for j in range(position[0], position[1] + 1):
                    character_positions.add(j)
            substructure_positions = set()
            not_substructure_positions = set()
            for character_position in character_positions:
                if substructure_atoms[character_position] == 1:
                    substructure_positions.add(character_position)
                else:
                    not_substructure_positions.add(character_position)
            substructure_values = cam[list(substructure_positions)]
            not_substructure_values = cam[list(not_substructure_positions)]
        else:
            for i in range(len(atom_locations)):
                if atom_locations[i, 0] >= 0:
                    position = (atom_locations[i, 0], atom_locations[i, 1])
                    if substructure_atoms[position] == 1:
                        substructure_values.append(cam[position])
                    else:
                        not_substructure_values.append(cam[position])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return len(not_substructure_values), numpy.mean(not_substructure_values), \
                   numpy.std(not_substructure_values), len(substructure_values), numpy.mean(substructure_values), \
                   numpy.std(substructure_values)

    @staticmethod
    def calculate_single_distance(smiles_string, substructure_atoms, cam, atom_locations=None):
        if atom_locations is None:
            character_positions = set()
            positions = smiles_analyzer.atom_positions(smiles_string)
            for position in positions:
                for j in range(position[0], position[1] + 1):
                    character_positions.add(j)
            differences = substructure_atoms[list(character_positions)] - cam[list(character_positions)]
        else:
            differences = list()
            for i in range(len(atom_locations)):
                position = (atom_locations[i, 0], atom_locations[i, 1])
                differences.append(substructure_atoms[position] - cam[position])
        return numpy.linalg.norm(differences)
