def calculate(molecule, atom_positions):
    bond_positions = dict()
    occupied_positions = set()
    for atom in molecule.GetAtoms():
        occupied_positions.add((atom_positions[atom.GetIdx()][0], atom_positions[atom.GetIdx()][1]))
    for bond in molecule.GetBonds():
        positions = list()
        x1 = atom_positions[bond.GetBeginAtomIdx()][0]
        y1 = atom_positions[bond.GetBeginAtomIdx()][1]
        x2 = atom_positions[bond.GetEndAtomIdx()][0]
        y2 = atom_positions[bond.GetEndAtomIdx()][1]
        f = LinearFunction(x1, y1, x2, y2)
        for x in range(min(x1, x2) + 1, max(x1, x2)):
            y = round(f.get_y(x))
            position = (x, y)
            if position not in occupied_positions:
                occupied_positions.add(position)
                positions.append(position)
        for y in range(min(y1, y2) + 1, max(y1, y2)):
            x = round(f.get_x(y))
            position = (x, y)
            if position not in occupied_positions:
                occupied_positions.add(position)
                positions.append(position)
        bond_positions[bond.GetIdx()] = positions
    return bond_positions


class LinearFunction:

    def __init__(self, x1, y1, x2, y2):
        if x1 - x2 == 0:
            self.x = x1
        else:
            self.x = None
            self.m = (y1 - y2) / (x1 - x2)
            self.b = y1 - self.m * x1

    def get_x(self, y):
        if self.x is not None:
            return self.x
        else:
            return (y - self.b) / self.m

    def get_y(self, x):
        return self.m * x + self.b
