from rdkit import Chem
from util import smiles_tokenizer


def atom_positions(smiles):
    positions = list()
    molecule = Chem.MolFromSmiles(smiles)
    smiles = smiles.lower()
    rest_index = 0
    for atom in molecule.GetAtoms():
        symbol = atom.GetSymbol().lower()
        start = smiles.find(symbol, rest_index)
        end = start + len(symbol) - 1
        positions.append([start, end])
        rest_index = end + 1
    return positions


def close_brackets(string):
    open_count = 0
    close_count = 0
    for character in string:
        if character == '(':
            open_count += 1
        elif character == ')':
            if open_count > 0:
                open_count -= 1
            else:
                close_count += 1
    for i in range(close_count):
        string = '(' + string
    for i in range(open_count):
        string += ')'
    return string


def split_separate_branches(smiles_string):
    smiles_strings = list()
    splits = 0
    index = 0
    while smiles_string[index] == '(':
        splits += 1
        index += 1
    if splits == 0:
        return [remove_unclosed_rings(smiles_string)]
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
                smiles_strings.append(remove_unclosed_rings(new_smiles))
                new_smiles = ''
                splits -= 1
        else:
            new_smiles += character
        index += 1
    smiles_strings.append(remove_unclosed_rings(smiles_string[index:]))
    need_resolving = list()
    for i in range(1, len(smiles_strings)):
        if smiles_strings[i].startswith('('):
            need_resolving.append(smiles_strings[i])
    for unfinished in need_resolving:
        smiles_strings.remove(unfinished)
        smiles_strings += split_separate_branches(unfinished)
    return smiles_strings


def replace_multi_use_ring_labels(smiles_string):
    tokens = smiles_tokenizer.TokenizedSmiles(smiles_string).get_tokens()
    label = 1
    new_smiles_string = ''
    open_rings = dict()
    for token in tokens:
        if token.get_type() == smiles_tokenizer.Token.RING:
            original_label = token.get_token()
            if original_label in open_rings:
                replacement_label = open_rings[original_label]
                del open_rings[original_label]
            else:
                replacement_label = str(label)
                label += 1
                if len(replacement_label) > 1:
                    replacement_label = '%' + replacement_label
                open_rings[original_label] = replacement_label
            new_smiles_string += replacement_label
        else:
            new_smiles_string += token.get_token()
    return new_smiles_string


def remove_unclosed_rings(smiles_string):
    label_pattern = smiles_tokenizer.TokenizedSmiles.ring_pattern
    parts = list()
    unclosed_rings = set()
    while len(smiles_string) > 0:
        match = label_pattern.match(smiles_string)
        if match is not None:
            length = match.span()[1]
            part = smiles_string[:length]
            if part in unclosed_rings:
                unclosed_rings.remove(part)
            else:
                unclosed_rings.add(part)
        else:
            length = 1
            part = smiles_string[:length]
        parts.append(part)
        smiles_string = smiles_string[length:]
    new_smiles_string = ''
    for part in parts:
        if part not in unclosed_rings:
            new_smiles_string += part
    return new_smiles_string


def clean_substructure(substructure_smiles_string):
    substructure_smiles_string = remove_unclosed_rings(substructure_smiles_string)
    substructure_smiles_string = close_brackets(substructure_smiles_string)
    substructure_smiles_strings = split_separate_branches(substructure_smiles_string)
    return substructure_smiles_strings
