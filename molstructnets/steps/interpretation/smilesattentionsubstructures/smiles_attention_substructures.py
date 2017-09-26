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
                substructures = concurrent_counting_set.ConcurrentCountingSet()
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
            first_token = tokens(token_index)
            token_index += 1
            # First token must be ATOM
            while first_token.get_type() != smiles_tokenizer.Token.ATOM:
                first_token = tokens(token_index)
                token_index += 1
            last_token = first_token
            # Add tokens until gap
            while last_token.get_position()[1] == tokens(token_index).get_position()[0]:
                last_token = tokens(token_index)
                token_index += 1
            backwards_index = token_index - 1
            # Remove trailing tokens if they are not ATOM or RING
            while last_token.get_type() != smiles_tokenizer.Token.ATOM or last_token.get_type() != smiles_tokenizer.Token.RING:
                backwards_index -= 1
                last_token = tokens(backwards_index)
            start = first_token.get_position()[0]
            end = last_token.get_position()[1]
            prefix_smiles = smiles_string[:start]
            substructure_smiles = smiles_string[start:end]
            substructure_smiles = SmilesAttentionSubstructures.remove_unclosed_rings(substructure_smiles, prefix_smiles)
            substructure_smiles = SmilesAttentionSubstructures.close_brackets(substructure_smiles)
            molecule = Chem.MolFromSmiles(substructure_smiles)
            substructure_smiles = Chem.MolToSmiles(molecule)
            if len(substructure_smiles) >= min_length:
                substructures_set.add(substructure_smiles)

    @staticmethod
    def remove_unclosed_rings(smiles_string, prefix_smiles_string):
        prefix_tokens = smiles_tokenizer.TokenizedSmiles(prefix_smiles_string)
        unclosed_prefix_rings = set()
        for token in prefix_tokens:
            if token.get_type() == smiles_tokenizer.Token.RING:
                label = token.get_token()
                if label in unclosed_prefix_rings:
                    unclosed_prefix_rings.remove(label)
                else:
                    unclosed_prefix_rings.add(label)
        smiles_tokens = smiles_tokenizer.TokenizedSmiles(smiles_string).get_tokens()
        positions_to_remove = list()
        unclosed_smiles_rings = set()
        for token in smiles_tokens:
            if token.get_type() == smiles_tokenizer.Token.RING:
                label = token.get_token()
                if label in unclosed_prefix_rings:
                    positions_to_remove.append(token.get_position())
                    unclosed_prefix_rings.remove(label)
                else:
                    if label in unclosed_prefix_rings:
                        unclosed_smiles_rings.remove(label)
                    else:
                        unclosed_smiles_rings.add(label)
        for token in smiles_tokens[::-1]:
            if token.get_type() == smiles_tokenizer.Token.RING:
                label = token.get_token()
                if label in unclosed_smiles_rings:
                    positions_to_remove.append(token.get_position())
                    unclosed_smiles_rings.remove(label)
        return misc.substring_cut_from_middle(smiles_string, positions_to_remove)

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
