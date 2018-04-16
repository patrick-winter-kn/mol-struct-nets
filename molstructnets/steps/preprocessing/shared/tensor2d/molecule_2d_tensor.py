import numpy
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType

from steps.preprocessing.shared.tensor2d import bond_positions
from steps.preprocessing.shared.chemicalproperties import chemical_properties

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


def molecule_to_2d_tensor(molecule, index_lookup, rasterizer_, preprocessed_shape, atom_locations_shape=None,
                          transformer_=None, random_=None, flip=False, rotation=0, with_chemical_properties=False):
    # We redo this if the transformation size does not fit
    while True:
        try:
            data_type = 'int16'
            if with_chemical_properties:
                data_type = 'float'
            preprocessed_row = numpy.zeros((preprocessed_shape[1], preprocessed_shape[2], preprocessed_shape[3]),
                                           dtype=data_type)
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
                position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
                x = position.x
                y = position.y
                if transformer_ is not None:
                    x, y = transformer_.apply(x, y, flip, rotation)
                x, y = rasterizer_.apply(x, y)
                # Check if coordinates fit into the shape
                if (not 0 <= x < preprocessed_row.shape[0]) or (not 0 <= y < preprocessed_row.shape[1]):
                    # Redo everything hoping for a better fitting transformation
                    raise ValueError()
                if atom.GetSymbol() in index_lookup:
                    symbol_index = index_lookup[atom.GetSymbol()]
                    preprocessed_row[x, y, symbol_index] = 1
                if with_chemical_properties:
                    preprocessed_row[x, y, len(index_lookup):] = chemical_properties.get_chemical_properties(atom)[:]
                if atom_locations_row is not None:
                    atom_locations_row[atom.GetIdx(), 0] = x
                    atom_locations_row[atom.GetIdx(), 1] = y
                atom_positions[atom.GetIdx()] = [x, y]
            bond_positions_ = bond_positions.calculate(molecule, atom_positions)
            for bond in molecule.GetBonds():
                bond_symbol = get_bond_symbol(bond.GetBondType())
                if bond_symbol is not None and bond_symbol in index_lookup:
                    bond_symbol_index = index_lookup[bond_symbol]
                    for position in bond_positions_[bond.GetIdx()]:
                        preprocessed_row[position[0], position[1], bond_symbol_index] = 1
            if with_empty_bits and ' ' in index_lookup:
                set_empty_bits(preprocessed_row, len(index_lookup), index_lookup[' '])
            return preprocessed_row, atom_locations_row
        except ValueError:
            # Redo everything hoping for a better fitting transformation
            pass


def set_empty_bits(preprocessed_row, number_symbols, empty_symbol_index):
    for x in range(preprocessed_row.shape[0]):
        for y in range(preprocessed_row.shape[1]):
            value_sum = 0
            for symbol in range(number_symbols):
                value_sum += preprocessed_row[x, y, symbol]
            if value_sum == 0:
                preprocessed_row[x, y, empty_symbol_index] = 1
