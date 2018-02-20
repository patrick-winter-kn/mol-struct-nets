import h5py
from steps.interpretation.extractattentionsubstructures import substructure_set
from util import data_validation, file_structure, file_util, progressbar, logger, misc, thread_pool, hdf5_util,\
    smiles_analyzer, constants
from rdkit import Chem
import numpy


number_threads = thread_pool.default_number_threads


class ExtractAttentionSubstructures:

    @staticmethod
    def get_id():
        return 'extract_attention_substructures'

    @staticmethod
    def get_name():
        return 'Extract Attention Substructures'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'threshold', 'name': 'Threshold', 'type': float, 'default': 0.25, 'min': 0.0,
                           'max': 1.0, 'description': 'The threshold used to decide which parts of the attention map'
                                                      ' are interpreted as part of the substructure.'})
        parameters.append({'id': 'partition', 'name': 'Partition', 'type': str, 'default': 'both',
                           'options': ['train', 'test', 'both'],
                           'description': 'The partition that the substructures will be extracted from. Options are:'
                                          ' train, test or both partitions. Default: both'})
        parameters.append({'id': 'weighted', 'name': 'Weighted by Probability', 'type': bool, 'default': False,
                           'description': 'If enabled the activation value will be weighted by the probability before'
                                          ' the threshold is applied. Default: False'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_preprocessed(global_parameters)
        data_validation.validate_attention_map(global_parameters)
        if local_parameters['partition'] != 'both':
            data_validation.validate_partition(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        attention_substructures_path = file_util.resolve_subpath(
            file_structure.get_interpretation_folder(global_parameters), 'attention_substructures.h5')
        if file_util.file_exists(attention_substructures_path):
            logger.log('Skipping step: ' + attention_substructures_path + ' already exists')
        else:
            attention_map_h5 = h5py.File(file_structure.get_attentionmap_file(global_parameters), 'r')
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            if local_parameters['weighted']:
                predictions_h5 = h5py.File(file_structure.get_prediction_file(global_parameters), 'r')
                predictions = predictions_h5[file_structure.Predictions.prediction]
            else:
                predictions = None
            preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data], 'r')
            if file_structure.Preprocessed.atom_locations in preprocessed_h5.keys():
                atom_locations = preprocessed_h5[file_structure.Preprocessed.atom_locations]
            else:
                atom_locations = None
            smiles = data_h5[file_structure.DataSet.smiles]
            temp_attention_substructures_path = file_util.get_temporary_file_path(
                'attention_substructures')
            attention_substructures_h5 = h5py.File(temp_attention_substructures_path, 'w')
            if file_structure.AttentionMap.attention_map_active in attention_map_h5.keys():
                ExtractAttentionSubstructures.extract_attention_map_substructures(
                    attention_map_h5, smiles, global_parameters, local_parameters, attention_substructures_h5,
                    True, atom_locations, predictions)
            if file_structure.AttentionMap.attention_map_inactive in attention_map_h5.keys():
                ExtractAttentionSubstructures.extract_attention_map_substructures(
                    attention_map_h5, smiles, global_parameters, local_parameters, attention_substructures_h5,
                    False, atom_locations, predictions)
            attention_substructures_h5.close()
            file_util.move_file(temp_attention_substructures_path, attention_substructures_path)
            attention_map_h5.close()
            data_h5.close()
            if predictions_h5 is not None:
                predictions_h5.close()

    @staticmethod
    def extract_attention_map_substructures(attention_map_h5, smiles, global_parameters, local_parameters,
                                            attention_substructures_h5, active, atom_locations=None, predictions=None):
        if active:
            log_message = 'Extracting active attention substructures'
            attention_map_dataset_name = file_structure.AttentionMap.attention_map_active
            attention_map_indices_dataset_name = file_structure.AttentionMap.attention_map_active_indices
            substructures_dataset_name = 'active_substructures'
            substructures_occurrences_dataset_name = 'active_substructures_occurrences'
            substructures_value_dataset_name = 'active_substructures_value'
            substructures_number_heavy_atoms_dataset_name = 'active_substructures_number_heavy_atoms'
            substructures_score_dataset_name = 'active_substructures_score'
        else:
            log_message = 'Extracting inactive attention substructures'
            attention_map_dataset_name = file_structure.AttentionMap.attention_map_inactive
            attention_map_indices_dataset_name = file_structure.AttentionMap.attention_map_inactive_indices
            substructures_dataset_name = 'inactive_substructures'
            substructures_occurrences_dataset_name = 'inactive_substructures_occurrences'
            substructures_value_dataset_name = 'inactive_substructures_value'
            substructures_number_heavy_atoms_dataset_name = 'inactive_substructures_number_heavy_atoms'
            substructures_score_dataset_name = 'inactive_substructures_score'
        attention_map = attention_map_h5[attention_map_dataset_name]
        if attention_map_indices_dataset_name in attention_map_h5.keys():
            indices = attention_map_h5[attention_map_indices_dataset_name]
        else:
            indices = range(len(attention_map))
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
        chunks = misc.chunk(len(smiles), number_threads)
        if predictions is not None:
            if active:
                probabilities = predictions[:,0]
            else:
                probabilities = predictions[:,1]
        else:
            probabilities = None
        with progressbar.ProgressBar(len(indices)) as progress:
            with thread_pool.ThreadPool(number_threads) as pool:
                for chunk in chunks:
                    pool.submit(ExtractAttentionSubstructures.extract, attention_map, indices, smiles, substructures,
                                local_parameters['threshold'], chunk['start'], chunk['end'], progress, atom_locations,
                                probabilities)
                pool.wait()
        substructures_dict = substructures.get_dict()
        substructures = list(substructures_dict.keys())
        max_length = 0
        for smiles_string in substructures:
            max_length = max(max_length, len(smiles_string))
        dtype = 'S' + str(max(max_length, 1))
        substructures_dataset = hdf5_util.create_dataset(attention_substructures_h5, substructures_dataset_name,
                                                         (len(substructures),), dtype=dtype)
        substructures_occurrences_dataset = hdf5_util.create_dataset(attention_substructures_h5,
                                                                     substructures_occurrences_dataset_name,
                                                                     (len(substructures),), dtype='I')
        substructures_value_dataset = hdf5_util.create_dataset(attention_substructures_h5,
                                                               substructures_value_dataset_name, (len(substructures),))
        substructures_number_heavy_atoms_dataset = hdf5_util.create_dataset(attention_substructures_h5,
                                                               substructures_number_heavy_atoms_dataset_name,
                                                               (len(substructures),), dtype='I')
        substructures_score_dataset = hdf5_util.create_dataset(attention_substructures_h5,
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
    def extract(attention_map, indices, smiles, substructures, threshold, start, end, progress, atom_locations=None,
                probabilities=None):
        for i in indices[start:end+1]:
            smiles_string = smiles[i].decode('utf-8')
            if atom_locations is not None:
                locations = atom_locations[i]
            else:
                locations = None
            if probabilities is not None:
                probability = probabilities[i]
            else:
                probability = None
            atom_indices, values = ExtractAttentionSubstructures.pick_atoms(attention_map[i], smiles_string, threshold,
                                                                            locations, probability)
            if len(atom_indices) > 0:
                molecule = Chem.MolFromSmiles(smiles_string)
                ExtractAttentionSubstructures.add_substructures(molecule, atom_indices, values, substructures)
            progress.increment()

    @staticmethod
    def pick_atoms_smiles(attention_map, smiles_string, threshold):
        atom_positions = smiles_analyzer.atom_positions(smiles_string)
        picked_indices = list()
        values = list()
        for i in range(len(atom_positions)):
            position = atom_positions[i]
            mean = 0
            position_range = range(position[0], position[1] + 1)
            for j in position_range:
                mean += attention_map[j]
            mean /= len(position_range)
            if mean >= threshold:
                picked_indices.append(i)
                values.append(mean)
        return picked_indices, values

    @staticmethod
    def pick_atoms(attention_map, smiles_string, threshold, atom_locations=None, probability=None):
        if atom_locations is None:
            return ExtractAttentionSubstructures.pick_atoms_smiles(attention_map, smiles_string, threshold)
        else:
            picked_indices = list()
            values = list()
            for i in range(len(atom_locations)):
                if atom_locations[i][0] < 0:
                    break
                location = tuple(atom_locations[i])
                value = attention_map[location]
                if probability is not None:
                    value *= probability
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
            value_mean = value_sum/len(indices)
            smiles = Chem.MolFragmentToSmiles(molecule, indices)
            substructure.add_substructure(smiles, value_mean)
