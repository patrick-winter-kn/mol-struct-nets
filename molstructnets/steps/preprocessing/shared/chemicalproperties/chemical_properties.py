import numpy
from rdkit import Chem
from rdkit.Chem import Crippen


class Properties:

    atomic_number = 'atomic_number'
    formal_charge = 'formal_charge'
    aromatic = 'aromatic'
    isotope = 'isotope'
    mass = 'mass'
    number_neighbors = 'number_neighbors'
    number_hs = 'number_hs'
    valence = 'valence'
    in_ring = 'in_ring'
    mol_log_p = 'mol_log_p'
    mol_mr = 'mol_mr'
    all = [atomic_number, formal_charge, aromatic, isotope, mass, number_neighbors, number_hs, valence, in_ring,
           mol_log_p, mol_mr]


def get_chemical_properties(atom, properties=None):
    if properties is None:
        properties = Properties.all
    values = numpy.zeros(len(properties))
    for i in range(len(properties)):
        if properties[i] == Properties.atomic_number:
            values[i] = atom.GetAtomicNum()
        if properties[i] == Properties.formal_charge:
            values[i] = atom.GetFormalCharge()
        if properties[i] == Properties.aromatic:
            values[i] = atom.GetIsAromatic()
        if properties[i] == Properties.isotope:
            values[i] = atom.GetIsotope()
        if properties[i] == Properties.mass:
            values[i] = atom.GetMass()
        if properties[i] == Properties.number_neighbors:
            values[i] = len(atom.GetNeighbors())
        if properties[i] == Properties.number_hs:
            values[i] = atom.GetTotalNumHs()
        if properties[i] == Properties.valence:
            values[i] = atom.GetTotalValence()
        if properties[i] == Properties.in_ring:
            values[i] = int(atom.IsInRing())
        if properties[i] == Properties.mol_log_p:
            values[i] = Crippen.MolLogP(Chem.MolFromSmiles(atom.GetSymbol()))
        if properties[i] == Properties.mol_mr:
            values[i] = Crippen.MolMR(Chem.MolFromSmiles(atom.GetSymbol()))
    return values
