import numpy
from rdkit import Chem


def generate_substructure_features(smiles_data, substructures, progress=None):
    preprocessed = numpy.zeros((len(smiles_data), len(substructures)), dtype='uint16')
    for i in range(len(smiles_data)):
        smiles = smiles_data[i].decode('utf-8')
        molecule = Chem.MolFromSmiles(smiles)
        for j in range(len(substructures)):
            preprocessed[i, j] = len(molecule.GetSubstructMatches(substructures[j]))
        if progress is not None:
            progress.increment()
    if progress is not None:
        progress.finish()
    return preprocessed
