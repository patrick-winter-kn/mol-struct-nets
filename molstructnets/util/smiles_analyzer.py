from rdkit import Chem


def atom_positions(smiles):
    positions = list()
    molecule = Chem.MolFromSmiles(smiles)
    rest_index = 0
    for atom in molecule.GetAtoms():
        symbol = atom.GetSymbol()
        start = smiles.find(symbol, rest_index)
        end = start + len(symbol) - 1
        positions.append([start, end])
        rest_index = end + 1
    return positions
