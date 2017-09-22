import h5py

from util import data_validation, file_structure, file_util, progressbar, logger, misc, thread_pool, concurrent_set, hdf5_util
from rdkit import Chem
import re


number_threads = thread_pool.default_number_threads


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
        parameters.append({'id': 'threshold', 'name': 'Threshold', 'type': float,
                           'description': 'The threshold used to decide which parts of the attention map are'
                                          ' interpreted as part of the substructure.'})
        parameters.append({'id': 'min_length', 'name': 'Minimum length', 'type': int,
                           'description': 'The minimum length of a found substructure. If it is smaller it will not be'
                                          ' rejected.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_attention_map(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        smiles_attention_substructures_path = file_util.resolve_subpath(file_structure.get_interpretation_folder(global_parameters), 'smiles_attention_substructures.h5')
        if file_util.file_exists(smiles_attention_substructures_path):
            logger.log('Skipping step: ' + smiles_attention_substructures_path + ' already exists')
        else:
            attention_map_h5 = h5py.File(file_structure.get_attentionmap_file(global_parameters), 'r')
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles = data_h5[file_structure.DataSet.smiles]
            temp_smiles_attention_substructures_path = file_util.get_temporary_file_path('smiles_attention_substructures')
            smiles_attention_substructures_h5 = h5py.File(temp_smiles_attention_substructures_path, 'w')
            if file_structure.AttentionMap.attention_map_active in attention_map_h5.keys():
                attention_map_active = attention_map_h5[file_structure.AttentionMap.attention_map_active]
                if file_structure.AttentionMap.attention_map_active_indices in attention_map_h5.keys():
                    indices = attention_map_h5[file_structure.AttentionMap.attention_map_active_indices]
                else:
                    indices = range(attention_map_active)
                logger.log('Extracting active SMILES attention substructures', logger.LogLevel.INFO)
                substructures = concurrent_set.ConcurrentSet()
                chunks = misc.chunk(len(smiles), number_threads)
                with progressbar.ProgressBar(len(indices)) as progress:
                    with thread_pool.ThreadPool(number_threads) as pool:
                        for chunk in chunks:
                            pool.submit(SmilesAttentionSubstructures.extract, attention_map_active, indices, smiles,
                                        substructures, local_parameters['threshold'], local_parameters['min_length'],
                                        chunk['start'], chunk['end'], progress)
                        pool.wait()
                substructures = list(substructures.get_set_copy())
                max_length = 0
                for smiles_string in substructures:
                    max_length = max(max_length, len(smiles_string))
                dtype = 'S' + str(max_length)
                active_substructures = hdf5_util.create_dataset(smiles_attention_substructures_h5, 'active_substructures', (len(substructures),), dtype=dtype)
                for i in range(len(substructures)):
                    active_substructures[i] = substructures[i].encode()
            if file_structure.AttentionMap.attention_map_inactive in attention_map_h5.keys():
                # TODO like active
                pass
            file_util.move_file(temp_smiles_attention_substructures_path, smiles_attention_substructures_path)
            attention_map_h5.close()
            data_h5.close()


    @staticmethod
    def extract(attention_map, indices, smiles, substructures_set, threshold, min_length, start, end, progress):
        for i in indices[start:end+1]:
            smiles_string = smiles[i].decode('utf-8')
            raw_strings = SmilesAttentionSubstructures.extract_raw_strings(smiles_string, attention_map[i], threshold)
            SmilesAttentionSubstructures.raw_strings_to_smiles(substructures_set, raw_strings, min_length)
            progress.increment()

    @staticmethod
    def extract_raw_strings(smiles_string, attention_map, threshold):
        raw_strings = list()
        raw_string = ''
        for i in range(len(smiles_string)):
            if attention_map[i] >= threshold:
                raw_string += smiles_string[i]
            elif len(raw_string) > 0:
                raw_strings.append(raw_string)
        if len(raw_string) > 0:
            raw_strings.append(raw_string)
        return raw_strings

    @staticmethod
    def raw_strings_to_smiles(substructure_set, raw_strings, min_length):
        for raw_string in raw_strings:
            raw_string = SmilesAttentionSubstructures.remove_leading_characters(raw_string, '[()-=0-9]')
            raw_string = SmilesAttentionSubstructures.remove_trailing_characters(raw_string, '[()-=]')
            raw_string = SmilesAttentionSubstructures.remove_unclosed_rings(raw_string)
            raw_string = SmilesAttentionSubstructures.close_brackets(raw_string)
            molecule = Chem.MolFromSmiles(raw_string)
            smiles = Chem.MolToSmiles(molecule)
            if len(smiles) >= min_length:
                substructure_set.add(smiles)

    @staticmethod
    def remove_leading_characters(string, pattern):
        pattern = re.compile(pattern)
        while len(string) > 0 and pattern.match(string[0]):
            string = string[1:]
        return string

    @staticmethod
    def remove_trailing_characters(string, pattern):
        pattern = re.compile(pattern)
        while len(string) > 0 and pattern.match(string[-1]):
            string = string[:-1]
        return string

    @staticmethod
    def remove_unclosed_rings(string):
        # TODO implement (problem: ...CCC1CCCC1CCC1CC...)
        return string

    @staticmethod
    def close_brackets(string):
        open_count = 0
        close_count = 0
        for character in string:
            if character == '(':
                open_count += 1
            elif character == ')':
                close_count += 1
        if open_count > close_count:
            for i in range(open_count - close_count):
                string += ')'
        elif close_count > open_count:
            for i in range(close_count - open_count):
                string = '(' + string
        return string

    # TODO what about half atoms (e.g. l from Cl)?
    