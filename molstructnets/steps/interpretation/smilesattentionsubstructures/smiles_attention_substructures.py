import h5py

from util import data_validation, file_structure, file_util, progressbar, logger, misc, thread_pool, concurrent_counting_set, hdf5_util, smiles_tokenizer
from rdkit import Chem


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
        parameters.append({'id': 'min_length', 'name': 'Minimum length (default: 1)', 'type': int, 'default': 1,
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
                SmilesAttentionSubstructures.extract_attention_map_substructures(attention_map_h5, smiles, local_parameters, smiles_attention_substructures_h5, True)
            if file_structure.AttentionMap.attention_map_inactive in attention_map_h5.keys():
                SmilesAttentionSubstructures.extract_attention_map_substructures(attention_map_h5, smiles, local_parameters, smiles_attention_substructures_h5, False)
            smiles_attention_substructures_h5.close()
            file_util.move_file(temp_smiles_attention_substructures_path, smiles_attention_substructures_path)
            attention_map_h5.close()
            data_h5.close()

    @staticmethod
    def extract_attention_map_substructures(attention_map_h5, smiles, local_parameters, smiles_attention_substructures_h5, active):
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
            indices = range(attention_map)
        logger.log(log_message, logger.LogLevel.INFO)
        substructures = concurrent_counting_set.ConcurrentCountingSet()
        chunks = misc.chunk(len(smiles), number_threads)
        with progressbar.ProgressBar(len(indices)) as progress:
            with thread_pool.ThreadPool(number_threads) as pool:
                for chunk in chunks:
                    pool.submit(SmilesAttentionSubstructures.extract, attention_map, indices, smiles,
                                substructures, local_parameters['threshold'], local_parameters['min_length'],
                                chunk['start'], chunk['end'], progress)
                pool.wait()
        substructures_dict = substructures.get_dict_copy()
        substructures = sorted(substructures_dict.keys())
        max_length = 0
        for smiles_string in substructures:
            max_length = max(max_length, len(smiles_string))
        dtype = 'S' + str(max(max_length, 1))
        substructures_dataset = hdf5_util.create_dataset(smiles_attention_substructures_h5, substructures_dataset_name, (len(substructures),), dtype=dtype)
        substructures_occurrences_dataset = hdf5_util.create_dataset(smiles_attention_substructures_h5, substructures_occurrences_dataset_name, (len(substructures),), dtype='I')
        for i in range(len(substructures)):
            substructures_dataset[i] = substructures[i].encode()
            substructures_occurrences_dataset[i] = substructures_dict[substructures[i]]

    @staticmethod
    def extract(attention_map, indices, smiles, substructures_set, threshold, min_length, start, end, progress):
        for i in indices[start:end+1]:
            smiles_string = smiles[i].decode('utf-8')
            tokenized_smiles = smiles_tokenizer.TokenizedSmiles(smiles_string)
            tokens = SmilesAttentionSubstructures.pick_tokens(tokenized_smiles, attention_map[i], threshold)
            SmilesAttentionSubstructures.extract_clean_substructures(smiles_string, tokens, min_length, substructures_set)
            progress.increment()

    @staticmethod
    def pick_tokens(tokenized_smiles, attention_map, threshold):
        picked_tokens = list()
        for token in tokenized_smiles.get_tokens():
            attention = 0
            position = token.get_position()
            for i in range(position[0], position[1]):
                attention += attention_map[i]
            attention /= position[1]
            if attention >= threshold:
                picked_tokens.append(token)
        return picked_tokens

    @staticmethod
    def extract_clean_substructures(smiles_string, tokens, min_length, substructures_set):
        token_index = 0
        while token_index < len(tokens):
            # Collect connected tokens
            first_token = tokens[token_index]
            token_index += 1
            # First token must be ATOM
            while token_index < len(tokens) and first_token.get_type() != smiles_tokenizer.Token.ATOM:
                first_token = tokens[token_index]
                token_index += 1
            # If we ended the previous loop because we found no legit starting token we need to end here
            if first_token.get_type() != smiles_tokenizer.Token.ATOM:
                break
            last_token = first_token
            # Add tokens until gap
            while token_index < len(tokens) and last_token.get_position()[1] == tokens[token_index].get_position()[0]:
                last_token = tokens[token_index]
                token_index += 1
            backwards_index = token_index - 1
            last_token = tokens[backwards_index]
            # Remove trailing tokens if they are not ATOM or RING
            while last_token.get_type() != smiles_tokenizer.Token.ATOM and last_token.get_type() != smiles_tokenizer.Token.RING:
                backwards_index -= 1
                last_token = tokens[backwards_index]
            start = first_token.get_position()[0]
            end = last_token.get_position()[1]
            substructure_smiles = SmilesAttentionSubstructures.remove_unclosed_rings(smiles_string, start, end)
            substructure_smiles = SmilesAttentionSubstructures.close_brackets(substructure_smiles)
            substructure_smiles_strings = SmilesAttentionSubstructures.split_separate_branches(substructure_smiles)
            for substructure_smiles_string in substructure_smiles_strings:
                molecule = Chem.MolFromSmiles(substructure_smiles_string)
                substructure_smiles_string = Chem.MolToSmiles(molecule)
                if len(substructure_smiles_string) >= min_length:
                    substructures_set.add(substructure_smiles_string)

    @staticmethod
    def remove_unclosed_rings(smiles_string, start, end):
        tokenized_smiles = smiles_tokenizer.TokenizedSmiles(smiles_string)
        prefix_tokens = tokenized_smiles.get_substring_tokens(0, start)
        smiles_tokens = tokenized_smiles.get_substring_tokens(start, end)
        unclosed_prefix_rings = set()
        for token in prefix_tokens:
            if token.get_type() == smiles_tokenizer.Token.RING:
                label = token.get_token()
                if label in unclosed_prefix_rings:
                    unclosed_prefix_rings.remove(label)
                else:
                    unclosed_prefix_rings.add(label)
        positions_to_remove = list()
        unclosed_smiles_rings = set()
        for token in smiles_tokens:
            if token.get_type() == smiles_tokenizer.Token.RING:
                label = token.get_token()
                if label in unclosed_prefix_rings:
                    positions_to_remove.append(token.get_position())
                    unclosed_prefix_rings.remove(label)
                else:
                    if label in unclosed_smiles_rings:
                        unclosed_smiles_rings.remove(label)
                    else:
                        unclosed_smiles_rings.add(label)
        for token in smiles_tokens[::-1]:
            if token.get_type() == smiles_tokenizer.Token.RING:
                label = token.get_token()
                if label in unclosed_smiles_rings:
                    positions_to_remove.append(token.get_position())
                    unclosed_smiles_rings.remove(label)
        return misc.substring_cut_from_middle(smiles_string, [(0,start)] + positions_to_remove + [(end, len(smiles_string))])

    @staticmethod
    def close_brackets(string):
        # TODO no brackets are added for e.g. C)C(C
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

    @staticmethod
    def split_separate_branches(smiles_string):
        smiles_strings = list()
        splits = 0
        index = 0
        while smiles_string[index] == '(':
            splits += 1
            index += 1
        if splits == 0:
            return [smiles_string]
        new_smiles = ''
        # We skip the starting branch brackets
        index = splits
        level = 0
        while splits > 0:
            character = smiles_string[index]
            if character == '(':
                level += 1
                new_smiles += character
            elif character == ')':
                if level > 0:
                    level -= 1
                    new_smiles += character
                else:
                    smiles_strings.append(new_smiles)
                    new_smiles = ''
                    splits -= 1
            else:
                new_smiles += character
            index += 1
        smiles_strings.append(smiles_string[index:])
        need_resolving = list()
        for i in range(1, len(smiles_strings)):
            if smiles_strings[i].startswith('('):
                need_resolving.append(smiles_strings[i])
        for unfinished in need_resolving:
            smiles_strings.remove(unfinished)
            smiles_strings += SmilesAttentionSubstructures.split_separate_branches(unfinished)
        return smiles_strings
