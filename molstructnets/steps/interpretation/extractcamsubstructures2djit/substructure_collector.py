from rdkit import Chem

from util import thread_pool


class SubstructureCollector:

    def __init__(self):
        self.substructures = dict()
        self.pool = thread_pool.ThreadPool(1)

    def collect(self, substructure_queue):
        self.pool.submit(self._collect, substructure_queue)

    def _collect(self, substructure_queue):
        while True:
            substructure = substructure_queue.get()
            if substructure is None:
                break
            self.add_substructure(substructure[0], substructure[1])

    def add_substructure(self, smiles, value):
        if smiles in self.substructures:
            self.substructures[smiles].add_occurrence(value)
        else:
            self.substructures[smiles] = Substructure(smiles, value)

    def get_dict(self):
        return self.substructures

    def close(self):
        self.pool.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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
