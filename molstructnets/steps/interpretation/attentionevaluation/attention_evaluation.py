from util import data_validation, file_structure, file_util, logger, hdf5_util, smiles_analyzer, progressbar, constants
import h5py
import numpy
import warnings


class AttentionEvaluation:

    @staticmethod
    def get_id():
        return 'attention_evaluation'

    @staticmethod
    def get_name():
        return 'Attention Evaluation'

    @staticmethod
    def get_parameters():
        parameters = list()
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_preprocessed(global_parameters)
        data_validation.validate_attention_map(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        attention_evaluation_active_path = file_util.resolve_subpath(
            file_structure.get_interpretation_folder(global_parameters), 'attention_evaluation_active.h5')
        attention_evaluation_inactive_path = file_util.resolve_subpath(
            file_structure.get_interpretation_folder(global_parameters), 'attention_evaluation_inactive.h5')
        attention_map_h5 = h5py.File(file_structure.get_attentionmap_file(global_parameters), 'r')
        preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data])
        atom_locations = None
        if file_structure.Preprocessed.atom_locations in preprocessed_h5.keys():
            atom_locations = preprocessed_h5[file_structure.Preprocessed.atom_locations]
        substructure_atoms = attention_map_h5[file_structure.AttentionMap.substructure_atoms]
        data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
        smiles = data_h5[file_structure.DataSet.smiles]
        if file_structure.AttentionMap.attention_map_active in attention_map_h5.keys():
            if file_util.file_exists(attention_evaluation_active_path):
                logger.log('Skipping: ' + attention_evaluation_active_path + ' already exists')
            else:
                indices = None
                if file_structure.AttentionMap.attention_map_active_indices in attention_map_h5.keys():
                    indices = attention_map_h5[file_structure.AttentionMap.attention_map_active_indices]
                temp_attention_evaluation_active_path = file_util.get_temporary_file_path('attention_evaluation_active')
                attention_map_active = attention_map_h5[file_structure.AttentionMap.attention_map_active]
                AttentionEvaluation.calculate_attention_evaluation(attention_map_active, substructure_atoms, smiles,
                                                                   temp_attention_evaluation_active_path, indices,
                                                                   atom_locations)
                file_util.move_file(temp_attention_evaluation_active_path, attention_evaluation_active_path)
        if file_structure.AttentionMap.attention_map_inactive in attention_map_h5.keys():
            if file_util.file_exists(attention_evaluation_inactive_path):
                logger.log('Skipping: ' + attention_evaluation_inactive_path + ' already exists')
            else:
                indices = None
                if file_structure.AttentionMap.attention_map_inactive_indices in attention_map_h5.keys():
                    indices = attention_map_h5[file_structure.AttentionMap.attention_map_inactive_indices]
                temp_attention_evaluation_inactive_path = file_util.get_temporary_file_path(
                    'attention_evaluation_inactive')
                attention_map_inactive = attention_map_h5[file_structure.AttentionMap.attention_map_inactive]
                AttentionEvaluation.calculate_attention_evaluation(attention_map_inactive, substructure_atoms, smiles,
                                                                   temp_attention_evaluation_inactive_path, indices,
                                                                   atom_locations)
                file_util.move_file(temp_attention_evaluation_inactive_path, attention_evaluation_inactive_path)
        data_h5.close()
        attention_map_h5.close()
        preprocessed_h5.close()

    @staticmethod
    def calculate_attention_evaluation(attention_map, substructure_atoms, smiles, attention_evaluation_path, indices,
                                       atom_locations=None):
        if indices is None:
            indices = range(len(attention_map))
        attention_evaluation_h5 = h5py.File(attention_evaluation_path, 'w')
        characters = hdf5_util.create_dataset(attention_evaluation_h5, 'characters', (len(indices),), dtype='I')
        mean = hdf5_util.create_dataset(attention_evaluation_h5, 'mean', (len(indices),))
        std_deviation = hdf5_util.create_dataset(attention_evaluation_h5, 'std_deviation', (len(indices),))
        substructure_characters = hdf5_util.create_dataset(attention_evaluation_h5, 'substructure_characters',
                                                           (len(indices),), dtype='I')
        substructure_mean = hdf5_util.create_dataset(attention_evaluation_h5, 'substructure_mean', (len(indices),))
        substructure_std_deviation = hdf5_util.create_dataset(attention_evaluation_h5, 'substructure_std_deviation',
                                                              (len(indices),))
        distance = hdf5_util.create_dataset(attention_evaluation_h5, 'distance', (len(indices),))
        index = 0
        with progressbar.ProgressBar(len(indices)) as progress:
            for i in indices:
                locations = None
                if atom_locations is not None:
                    locations = atom_locations[i]
                characters[index], mean[index], std_deviation[index], substructure_characters[index],\
                substructure_mean[index], substructure_std_deviation[index] =\
                    AttentionEvaluation.calculate_single_attention_evaluation(smiles[i].decode('utf-8'),
                                                                              substructure_atoms[i],
                                                                              attention_map[i], locations)
                distance[index] = AttentionEvaluation.calculate_single_distance(smiles[i].decode('utf-8'),
                                                                                substructure_atoms[i], attention_map[i],
                                                                                locations)
                index += 1
                progress.increment()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            overall_mean = numpy.mean(mean)
            overall_substructure_mean = numpy.mean(substructure_mean)
            overall_mean_std_deviation = numpy.std(mean)
            overall_substructure_mean_std_deviation = numpy.std(substructure_mean)
        attention_evaluation_h5.close()
        hdf5_util.set_property(attention_evaluation_path, 'overall_mean', overall_mean)
        hdf5_util.set_property(attention_evaluation_path, 'overall_substructure_mean', overall_substructure_mean)
        hdf5_util.set_property(attention_evaluation_path, 'overall_mean_std_deviation', overall_mean_std_deviation)
        hdf5_util.set_property(attention_evaluation_path, 'overall_substructure_mean_std_deviation',
                               overall_substructure_mean_std_deviation)

    @staticmethod
    def calculate_single_attention_evaluation(smiles_string, substructure_atoms, attention_map, atom_locations=None):
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
            substructure_values = attention_map[list(substructure_positions)]
            not_substructure_values = attention_map[list(not_substructure_positions)]
        else:
            for i in range(len(atom_locations)):
                if atom_locations[i, 0] >= 0:
                    position = (atom_locations[i, 0], atom_locations[i, 1])
                    if substructure_atoms[position] == 1:
                        substructure_values.append(attention_map[position])
                    else:
                        not_substructure_values.append(attention_map[position])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return len(not_substructure_values), numpy.mean(not_substructure_values),\
                   numpy.std(not_substructure_values), len(substructure_values), numpy.mean(substructure_values),\
                   numpy.std(substructure_values)

    @staticmethod
    def calculate_single_distance(smiles_string, substructure_atoms, attention_map, atom_locations=None):
        if atom_locations is None:
            character_positions = set()
            positions = smiles_analyzer.atom_positions(smiles_string)
            for position in positions:
                for j in range(position[0], position[1] + 1):
                    character_positions.add(j)
            differences = substructure_atoms[list(character_positions)] - attention_map[list(character_positions)]
        else:
            differences = list()
            for i in range(len(atom_locations)):
                position = (atom_locations[i, 0], atom_locations[i, 1])
                differences.append(substructure_atoms[position] - attention_map[position])
        return numpy.linalg.norm(differences)
