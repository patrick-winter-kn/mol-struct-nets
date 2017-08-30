import re
import random
from rdkit import Chem


atoms = {'O': 0.13, 'S': 0.03, 'N': 0.12, 'C': 0.72}
branches = 0.19
rings = 0.14
atom_pattern = re.compile('[A-Z]')


class SmilesGenerator:

    def __init__(self, number, max_length, seed, offset=0, progress=None, global_smiles_set=None):
        self.number = number
        self.max_length = max_length
        self.random = random.Random()
        self.random.seed(seed + offset)
        self.offset = offset
        self.progress = progress
        self.global_smiles_set = global_smiles_set

    def is_branch(self, action, current_length, target_length):
        return action <= branches and current_length > 0 and current_length + 4 < target_length


    def is_ring(self, action, current_length, target_length, label_size):
        if label_size > 1:
            label_size += 1
        return action > branches and action <= rings + branches and current_length > 0 and current_length + label_size * 2 + 2 < target_length


    def pick_atom(self):
        rest = random.uniform(0.0, 1.0)
        for atom in atoms:
            rest -= atoms[atom]
            if rest <= 0:
                return atom


    def is_atom(self, char):
        global atom_pattern
        return atom_pattern.match(char)


    def generate_single_smiles(self, min_length, max_length, is_in_ring, ring_label):
        length = random.randint(min_length,max_length)
        string = ''
        while len(string) < length:
            action = random.uniform(0.0, 1.0)
            if self.is_branch(action, len(string), length) and self.is_atom(string[-1]):
                string += '(' + self.generate_single_smiles(1, length - len(string) - 3, False, ring_label) + ')' + self.pick_atom()
            elif self.is_ring(action, len(string), length, len(str(ring_label))) and self.is_atom(string[-1]) and not is_in_ring:
                label = str(ring_label)
                if ring_label > 9:
                    label = '%' + label
                ring_label += 1
                string += label + self.generate_single_smiles(2, length - len(string) - (len(label) * 2), True, ring_label) + label
            else:
                string += self.pick_atom()
        return string

    def write_smiles(self, array):
        for i in range(self.offset, self.number + self.offset):
            valid = False
            while not valid:
                smiles = self.generate_single_smiles(1, self.max_length, False, 1).encode()
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles, sanitize=False))
                if len(smiles) <= self.max_length:
                    valid = self.global_smiles_set is None or self.global_smiles_set.add(smiles)
            array[i] = smiles.encode('utf-8')
            self.progress.increment()
