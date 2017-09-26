import re
from rdkit import Chem


class TokenizedSmiles:

    atom_pattern_string = 'SYMBOL|\\[SYMBOL(@|@@)?H?(([0-9]+([+]+|[-]+))|(([+]+|[-]+)[0-9]*))?\\]'
    branch_pattern = re.compile('[()]')
    ring_pattern = re.compile('[0-9]|%[0-9][0-9]+')
    bond_pattern = re.compile('[.\\-=#$:/\\\\]')

    def __init__(self, smiles_string):
        self.smiles_string = smiles_string
        self.tokens = list()
        self.tokenize()

    def tokenize(self):
        atom_symbols = list()
        molecule = Chem.MolFromSmiles(self.smiles_string)
        for atom in molecule.GetAtoms():
            atom_symbols.append(atom.GetSymbol())
        rest_string = self.smiles_string
        atom_index = -1
        done = 0
        type = Token.ATOM
        while len(rest_string) > 0:
            if type == Token.ATOM:
                atom_index += 1
                if atom_index < len(atom_symbols):
                    atom_pattern = re.compile(TokenizedSmiles.atom_pattern_string.replace('SYMBOL', atom_symbols[atom_index]), re.IGNORECASE)
            match, type = self.matching_pattern(rest_string, atom_pattern)
            length = match.span()[1]
            self.tokens.append(Token(self.smiles_string, done, done + length, type))
            done += length
            rest_string = rest_string[length:]

    def matching_pattern(self, string, atom_pattern):
        if atom_pattern is not None:
            atom_match = atom_pattern.match(string)
            if atom_match:
                return atom_match, Token.ATOM
        branch_match = TokenizedSmiles.branch_pattern.match(string)
        if branch_match:
            return branch_match, Token.BRANCH
        ring_match = TokenizedSmiles.ring_pattern.match(string)
        if ring_match:
            return ring_match, Token.RING
        bond_match = TokenizedSmiles.bond_pattern.match(string)
        if bond_match:
            return bond_match, Token.BOND
        return None, None

    def get_tokens(self):
        return self.tokens

    def get_substring_tokens(self, start, end):
        substring_tokens = list()
        for token in self.tokens:
            position = token.get_position()
            if position[0] >= start and position[1] <= end:
                substring_tokens.append(token)
        return substring_tokens


class Token:

    ATOM = 'atom'
    BOND = 'bond'
    BRANCH = 'branch'
    RING = 'ring'

    def __init__(self, smiles, start, end, type):
        self.smiles = smiles
        self.start = start
        self.end = end
        self.type = type

    def get_position(self):
        return (self.start, self.end)

    def get_type(self):
        return self.type

    def get_token(self):
        return self.smiles[self.start:self.end]

    def __str__(self):
        return self.get_token()

    def __repr__(self):
        return self.get_token()
