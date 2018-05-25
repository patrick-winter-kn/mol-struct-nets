import numpy
from rdkit.Chem import rdMolDescriptors, rdPartialCharges


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
    partial_charge = 'partial_charge'
    asa = 'asa'
    all = [atomic_number, formal_charge, aromatic, isotope, mass, number_neighbors, number_hs, valence, in_ring,
           mol_log_p, mol_mr, partial_charge, asa]
    selected = [formal_charge, aromatic, number_neighbors, valence, in_ring, mol_log_p, mol_mr, partial_charge, asa]


def get_chemical_properties(atom, properties=None):
    if properties is None:
        properties = Properties.selected
    values = numpy.zeros(len(properties), dtype='float32')
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
            values[i] = rdMolDescriptors._CalcCrippenContribs(atom.GetOwningMol())[atom.GetIdx()][0]
        if properties[i] == Properties.mol_mr:
            values[i] = rdMolDescriptors._CalcCrippenContribs(atom.GetOwningMol())[atom.GetIdx()][1]
        if properties[i] == Properties.partial_charge:
            if atom.HasProp('_GasteigerCharge') < 1:
                rdPartialCharges.ComputeGasteigerCharges(atom.GetOwningMol())
            values[i] == atom.GetDoubleProp('_GasteigerCharge')
        if properties[i] == Properties.asa:
            values[i] == list(rdMolDescriptors._CalcLabuteASAContribs(atom.GetOwningMol())[0])[atom.GetIdx()]
    return values
