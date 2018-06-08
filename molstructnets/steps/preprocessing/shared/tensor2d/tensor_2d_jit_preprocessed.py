class Tensor2DJitPreprocessed():

    def __init__(self):
        self._atoms = list()

    def add_atom(self, atom):
        self._atoms.append(atom)

    def fill_array(self, array):
        for atom in self._atoms:
            atom.fill_array(array)


class Tensor2DJitPreprocessedAtom():

    def __init__(self, position_x, position_y, symbol=None, features=None):
        self._position_x = position_x
        self._position_y = position_y
        self._symbol = symbol
        self._features = features

    def fill_array(self, array):
        if self._symbol is not None:
            array[self._position_x, self._position_y, self._symbol] = 1
        if self._features is not None:
            array[self._position_x, self._position_y, -len(self._features):] = self._features[:]
