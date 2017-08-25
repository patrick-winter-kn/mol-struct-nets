import re
import random


atoms = {'o': 0.13, 's': 0.03, 'n': 0.12, 'c': 0.72}
branches = 0.19
rings = 0.14
atom_pattern = re.compile('[a-z]')


class SmilesGenerator:

    def __init__(self, number, max_length, seed, offset=0, progress=None, duplicate_checker=None):
        self.number = number
        self.max_length = max_length
        self.random = random.Random()
        self.random.seed(seed + offset)
        self.offset = offset
        self.progress = progress
        self.duplicate_checker = duplicate_checker

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
            invalid = True
            while invalid:
                smiles = self.generate_single_smiles(1, self.max_length, False, 1).encode()
                invalid = self.duplicate_checker.is_duplicate(smiles)
            array[i] = smiles
            self.progress.increment()
