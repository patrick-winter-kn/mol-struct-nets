import numpy
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType

from steps.preprocessing.shared.matrix2d import bond_positions

with_empty_bits = False
padding = 2


def get_bond_symbol(bond_type):
    if bond_type == BondType.ZERO:
        return None
    elif bond_type == BondType.SINGLE:
        return '-'
    elif bond_type == BondType.DOUBLE:
        return '='
    elif bond_type == BondType.TRIPLE:
        return '#'
    elif bond_type == BondType.QUADRUPLE:
        return '$'
    elif bond_type == BondType.AROMATIC:
        return ':'
    else:
        return '-'


def molecule_to_2d_matrix(molecule, index_lookup, rasterizer_, preprocessed_shape, atom_locations_shape=None,
                          transformer_=None, random_=None, flip=False, rotation=0):
    # We redo this if the transformation size does not fit
    while True:
        preprocessed_row = numpy.zeros((preprocessed_shape[1], preprocessed_shape[2], preprocessed_shape[3]),
                                       dtype='int16')
        if atom_locations_shape is None:
            atom_locations_row = None
        else:
            atom_locations_row = numpy.full((atom_locations_shape[1], atom_locations_shape[2]), -1, dtype='int16')
        atom_positions = dict()
        AllChem.Compute2DCoords(molecule)
        if transformer_ is not None and random_ is not None:
            flip = bool(random_.getrandbits(1))
            rotation = random_.randrange(0, 360)
        for atom in molecule.GetAtoms():
            symbol_index = index_lookup[atom.GetSymbol()]
            position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
            x = position.x
            y = position.y
            if transformer_ is not None:
                x, y = transformer_.apply(x, y, flip, rotation)
            x, y = rasterizer_.apply(x, y)
            # Check if coordinates fit into the shape
            if x >= preprocessed_row.shape[0] or y >= preprocessed_row.shape[1]:
                # Redo everything hoping for a better fitting transformation
                continue
            preprocessed_row[x, y, symbol_index] = 1
            if atom_locations_row is not None:
                atom_locations_row[atom.GetIdx(), 0] = x
                atom_locations_row[atom.GetIdx(), 1] = y
            atom_positions[atom.GetIdx()] = [x, y]
        bond_positions_ = bond_positions.calculate(molecule, atom_positions)
        for bond in molecule.GetBonds():
            bond_symbol = get_bond_symbol(bond.GetBondType())
            if bond_symbol is not None:
                bond_symbol_index = index_lookup[bond_symbol]
                for position in bond_positions_[bond.GetIdx()]:
                    preprocessed_row[position[0], position[1], bond_symbol_index] = 1
        if with_empty_bits:
            set_empty_bits(preprocessed_row, index_lookup[' '])
        return preprocessed_row, atom_locations_row


def set_empty_bits(preprocessed_row, empty_symbol_index):
    for x in range(preprocessed_row.shape[0]):
        for y in range(preprocessed_row.shape[1]):
            value_sum = 0
            for symbol in range(preprocessed_row.shape[2]):
                value_sum += preprocessed_row[x, y, symbol]
            if value_sum == 0:
                preprocessed_row[x, y, empty_symbol_index] = 1
