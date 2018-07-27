import threading

from rdkit import Chem


class SubstructureSet:

    def __init__(self):
        self.substructures = dict()
        self.lock = threading.Lock()

    def add_substructure(self, smiles, value):
        self.lock.acquire()
        if smiles in self.substructures:
            self.substructures[smiles].add_occurrence(value)
        else:
            self.substructures[smiles] = Substructure(smiles, value)
        self.lock.release()

    def get_dict(self):
        return self.substructures


class Substructure:

    def __init__(self, smiles, value):
        self.smiles = smiles
        self.value_sum = value
        self.occurrences = 1
        self.number_heavy_atoms = Chem.MolFromSmiles(smiles, sanitize=False).GetNumHeavyAtoms()

    def add_occurrence(self, value):
        self.occurrences += 1
        self.value_sum += value

    def get_smiles(self):
        return self.smiles

    def get_occurrences(self):
        return self.occurrences

    def get_mean_value(self):
        return self.value_sum / self.occurrences

    def get_number_heavy_atoms(self):
        return self.number_heavy_atoms
