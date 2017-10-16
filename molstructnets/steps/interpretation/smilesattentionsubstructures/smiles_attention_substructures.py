import h5py
from steps.interpretation.smilesattentionsubstructures import substructure_set
from util import data_validation, file_structure, file_util, progressbar, logger, misc, thread_pool, hdf5_util,\
    smiles_analyzer, constants
from rdkit import Chem
import numpy


number_threads = 1


class SmilesAttentionSubstructures:

    @staticmethod
    def get_id():
        return 'smiles_attention_substructures'

    @staticmethod
    def get_name():
        return 'SMILES Attention Substructures'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'threshold', 'name': 'Threshold', 'type': float, 'default': 0.25,
                           'description': 'The threshold used to decide which parts of the attention map are'
                                          ' interpreted as part of the substructure.'})
        parameters.append({'id': 'partition', 'name': 'Partition (options: train or test, default: both)', 'type': str,
                           'default': 'both',
                           'description': 'The partition that the substructures will be extracted from. By default both'
                                          ' the train and test partition will be used.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_attention_map(global_parameters)
        if local_parameters['partition'] != 'both':
            data_validation.validate_partition(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        smiles_attention_substructures_path = file_util.resolve_subpath(
            file_structure.get_interpretation_folder(global_parameters), 'smiles_attention_substructures.h5')
        if file_util.file_exists(smiles_attention_substructures_path):
            logger.log('Skipping step: ' + smiles_attention_substructures_path + ' already exists')
        else:
            attention_map_h5 = h5py.File(file_structure.get_attentionmap_file(global_parameters), 'r')
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles = data_h5[file_structure.DataSet.smiles]
            temp_smiles_attention_substructures_path = file_util.get_temporary_file_path(
                'smiles_attention_substructures')
            smiles_attention_substructures_h5 = h5py.File(temp_smiles_attention_substructures_path, 'w')
            if file_structure.AttentionMap.attention_map_active in attention_map_h5.keys():
                SmilesAttentionSubstructures.extract_attention_map_substructures(
                    attention_map_h5, smiles, global_parameters, local_parameters, smiles_attention_substructures_h5,
                    True)
            if file_structure.AttentionMap.attention_map_inactive in attention_map_h5.keys():
                SmilesAttentionSubstructures.extract_attention_map_substructures(
                    attention_map_h5, smiles, global_parameters, local_parameters, smiles_attention_substructures_h5,
                    False)
            smiles_attention_substructures_h5.close()
            file_util.move_file(temp_smiles_attention_substructures_path, smiles_attention_substructures_path)
            attention_map_h5.close()
            data_h5.close()

    @staticmethod
    def extract_attention_map_substructures(attention_map_h5, smiles, global_parameters, local_parameters,
                                            smiles_attention_substructures_h5, active):
        if active:
            log_message = 'Extracting active SMILES attention substructures'
            attention_map_dataset_name = file_structure.AttentionMap.attention_map_active
            attention_map_indices_dataset_name = file_structure.AttentionMap.attention_map_active_indices
            substructures_dataset_name = 'active_substructures'
            substructures_occurrences_dataset_name = 'active_substructures_occurrences'
            substructures_value_dataset_name = 'active_substructures_value'
            substructures_score_dataset_name = 'active_substructures_score'
        else:
            log_message = 'Extracting inactive SMILES attention substructures'
            attention_map_dataset_name = file_structure.AttentionMap.attention_map_inactive
            attention_map_indices_dataset_name = file_structure.AttentionMap.attention_map_inactive_indices
            substructures_dataset_name = 'inactive_substructures'
            substructures_occurrences_dataset_name = 'inactive_substructures_occurrences'
            substructures_value_dataset_name = 'inactive_substructures_value'
            substructures_score_dataset_name = 'inactive_substructures_score'
        attention_map = attention_map_h5[attention_map_dataset_name]
        if attention_map_indices_dataset_name in attention_map_h5.keys():
            indices = attention_map_h5[attention_map_indices_dataset_name]
        else:
            indices = range(len(attention_map))
        if local_parameters['partition'] != 'both':
            partition_h5 = h5py.File(global_parameters[constants.GlobalParameters.partition_data], 'r')
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
        with progressbar.ProgressBar(len(indices)) as progress:
            with thread_pool.ThreadPool(number_threads) as pool:
                for chunk in chunks:
                    pool.submit(SmilesAttentionSubstructures.extract, attention_map, indices, smiles,
                                substructures, local_parameters['threshold'], chunk['start'], chunk['end'], progress)
                pool.wait()
        substructures_dict = substructures.get_dict()
        substructures = sorted(substructures_dict.keys())
        max_length = 0
        for smiles_string in substructures:
            max_length = max(max_length, len(smiles_string))
        dtype = 'S' + str(max(max_length, 1))
        substructures_dataset = hdf5_util.create_dataset(smiles_attention_substructures_h5, substructures_dataset_name,
                                                         (len(substructures),), dtype=dtype)
        substructures_occurrences_dataset = hdf5_util.create_dataset(smiles_attention_substructures_h5,
                                                                     substructures_occurrences_dataset_name,
                                                                     (len(substructures),), dtype='I')
        substructures_value_dataset = hdf5_util.create_dataset(smiles_attention_substructures_h5,
                                                               substructures_value_dataset_name, (len(substructures),))
        substructures_score_dataset = hdf5_util.create_dataset(smiles_attention_substructures_h5,
                                                               substructures_score_dataset_name, (len(substructures),))
        for i in range(len(substructures)):
            substructures_dataset[i] = substructures[i].encode()
            substructure = substructures_dict[substructures[i]]
            substructures_occurrences_dataset[i] = substructure.get_occurrences()
            substructures_value_dataset[i] = substructure.get_mean_value()
            substructures_score_dataset[i] = substructure.get_score()

    @staticmethod
    def extract(attention_map, indices, smiles, substructures, threshold, start, end, progress):
        for i in indices[start:end+1]:
            smiles_string = smiles[i].decode('utf-8')
            atom_indices, values = SmilesAttentionSubstructures.pick_atoms(attention_map[i], smiles_string, threshold)
            if len(atom_indices) > 0:
                molecule = Chem.MolFromSmiles(smiles_string)
                SmilesAttentionSubstructures.add_substructures(molecule, atom_indices, values, substructures)
            progress.increment()

    @staticmethod
    def pick_atoms(attention_map, smiles_string, threshold):
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
