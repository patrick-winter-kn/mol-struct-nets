from rdkit.Chem.rdchem import BondType


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
