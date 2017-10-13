import h5py

from util import data_validation, file_structure, file_util, progressbar, logger, misc, thread_pool,\
    concurrent_counting_set, hdf5_util, smiles_analyzer
from rdkit import Chem


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
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_attention_map(global_parameters)

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
                    attention_map_h5, smiles, local_parameters, smiles_attention_substructures_h5, True)
            if file_structure.AttentionMap.attention_map_inactive in attention_map_h5.keys():
                SmilesAttentionSubstructures.extract_attention_map_substructures(
                    attention_map_h5, smiles, local_parameters, smiles_attention_substructures_h5, False)
            smiles_attention_substructures_h5.close()
            file_util.move_file(temp_smiles_attention_substructures_path, smiles_attention_substructures_path)
            attention_map_h5.close()
            data_h5.close()

    @staticmethod
    def extract_attention_map_substructures(attention_map_h5, smiles, local_parameters,
                                            smiles_attention_substructures_h5, active):
        if active:
            log_message = 'Extracting active SMILES attention substructures'
            attention_map_dataset_name = file_structure.AttentionMap.attention_map_active
            attention_map_indices_dataset_name = file_structure.AttentionMap.attention_map_active_indices
            substructures_dataset_name = 'active_substructures'
            substructures_occurrences_dataset_name = 'active_substructures_occurrences'
        else:
            log_message = 'Extracting inactive SMILES attention substructures'
            attention_map_dataset_name = file_structure.AttentionMap.attention_map_inactive
            attention_map_indices_dataset_name = file_structure.AttentionMap.attention_map_inactive_indices
            substructures_dataset_name = 'inactive_substructures'
            substructures_occurrences_dataset_name = 'inactive_substructures_occurrences'
        attention_map = attention_map_h5[attention_map_dataset_name]
        if attention_map_indices_dataset_name in attention_map_h5.keys():
            indices = attention_map_h5[attention_map_indices_dataset_name]
        else:
            indices = range(len(attention_map))
        logger.log(log_message, logger.LogLevel.INFO)
        substructures = concurrent_counting_set.ConcurrentCountingSet()
        chunks = misc.chunk(len(smiles), number_threads)
        with progressbar.ProgressBar(len(indices)) as progress:
            with thread_pool.ThreadPool(number_threads) as pool:
                for chunk in chunks:
                    pool.submit(SmilesAttentionSubstructures.extract, attention_map, indices, smiles,
                                substructures, local_parameters['threshold'], chunk['start'], chunk['end'], progress)
                pool.wait()
        substructures_dict = substructures.get_dict_copy()
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
        for i in range(len(substructures)):
            substructures_dataset[i] = substructures[i].encode()
            substructures_occurrences_dataset[i] = substructures_dict[substructures[i]]

    @staticmethod
    def extract(attention_map, indices, smiles, substructures_set, threshold, start, end, progress):
        for i in indices[start:end+1]:
            smiles_string = smiles[i].decode('utf-8')
            atom_indices = SmilesAttentionSubstructures.pick_atoms(attention_map[i], smiles_string, threshold)
            if len(atom_indices) > 0:
                molecule = Chem.MolFromSmiles(smiles_string)
                substructure_smiles = Chem.MolFragmentToSmiles(molecule, atom_indices)
                substructures_set.update(substructure_smiles.split('.'))
            progress.increment()

    @staticmethod
    def pick_atoms(attention_map, smiles_string, threshold):
        atom_positions = smiles_analyzer.atom_positions(smiles_string)
        picked_indices = set()
        for i in range(len(atom_positions)):
            position = atom_positions[i]
            mean = 0
            position_range = range(position[0], position[1] + 1)
            for j in position_range:
                mean += attention_map[j]
            mean /= len(position_range)
            if mean >= threshold:
                picked_indices.add(i)
        return picked_indices
