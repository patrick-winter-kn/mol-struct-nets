import re
import random
from rdkit import Chem


atom_ps = {'O': 0.13, 'S': 0.03, 'N': 0.12, 'C': 0.72}
atoms = sorted(atom_ps)
branches = 0.19
rings = 0.14
atom_pattern = re.compile('[A-Z]')


class SmilesGenerator:

    def __init__(self, number, max_length, seed, offset=0, progress=None, global_smiles_set=None):
        self.number = number
        self.max_length = max_length
        self.random = random.Random()
        self.seed = seed
        self.offset = offset
        self.progress = progress
        self.global_smiles_set = global_smiles_set

    @staticmethod
    def is_branch(action, current_length, target_length):
        return action <= branches and current_length > 0 and current_length + 4 < target_length

    @staticmethod
    def is_ring(action, current_length, target_length, label_size):
        if label_size > 1:
            label_size += 1
        enough_space = current_length + label_size * 2 + 2 < target_length
        return branches < action <= rings + branches and current_length > 0 and enough_space

    def pick_atom(self):
        rest = self.random.uniform(0.0, 1.0)
        for atom in atoms:
            rest -= atom_ps[atom]
            if rest <= 0:
                return atom

    @staticmethod
    def is_atom(char):
        global atom_pattern
        return atom_pattern.match(char)

    def generate_single_smiles(self, min_length, max_length, is_in_ring, ring_label):
        length = self.random.randint(min_length, max_length)
        string = ''
        while len(string) < length:
            action = self.random.uniform(0.0, 1.0)
            if SmilesGenerator.is_branch(action, len(string), length) and SmilesGenerator.is_atom(string[-1]):
                # branch
                string += '(' + self.generate_single_smiles(1, length - len(string) - 3, False, ring_label) + ')'\
                          + self.pick_atom()
            elif SmilesGenerator.is_ring(action, len(string), length, len(str(ring_label[0])))\
                    and SmilesGenerator.is_atom(string[-1]) and not is_in_ring:
                # ring
                label = str(ring_label[0])
                if ring_label[0] > 9:
                    label = '%' + label
                ring_label[0] += 1
                string += label + self.generate_single_smiles(2, length - len(string) - (len(label) * 2),
                                                              True, ring_label) + label
            else:
                # single atom
                string += self.pick_atom()
        return string

    def write_smiles(self, array):
        for i in range(self.offset, self.number + self.offset):
            self.random.seed(self.seed + i)
            smiles = None
            valid = False
            while not valid:
                smiles = self.generate_single_smiles(1, self.max_length, False, [1]).encode()
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=False))
                if len(smiles) <= self.max_length:
                    valid = self.global_smiles_set is None or self.global_smiles_set.add(smiles)
            array[i] = smiles.encode('utf-8')
            self.progress.increment()
