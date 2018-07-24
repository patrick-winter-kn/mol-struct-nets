import random
import numpy
import h5py
from util import hdf5_util, file_structure, normalization
from rdkit import Chem
from rdkit.Chem import AllChem
from steps.preprocessing.shared.chemicalproperties import chemical_properties
from steps.preprocessing.shared.tensor2d import rasterizer, bond_positions, bond_symbols, tensor_2d_jit_preprocessed
from steps.preprocessingtraining.tensor2dtransformation import transformer


padding = 2


class Tensor2DJitPreprocessor:

    def __init__(self, preprocessed_path):
        preprocessed_h5 = h5py.File(preprocessed_path, 'r')
        self._shape =\
            tuple(hdf5_util.get_property(preprocessed_h5, file_structure.PreprocessedTensor2DJit.dimensions))
        self._with_bonds = hdf5_util.get_property(preprocessed_h5, file_structure.PreprocessedTensor2DJit.with_bonds)
        self._scale = hdf5_util.get_property(preprocessed_h5, file_structure.PreprocessedTensor2DJit.scale)
        square = hdf5_util.get_property(preprocessed_h5, file_structure.PreprocessedTensor2DJit.square)
        min_x = hdf5_util.get_property(preprocessed_h5, file_structure.PreprocessedTensor2DJit.min_x)
        min_y = hdf5_util.get_property(preprocessed_h5, file_structure.PreprocessedTensor2DJit.min_y)
        max_x = hdf5_util.get_property(preprocessed_h5, file_structure.PreprocessedTensor2DJit.max_x)
        max_y = hdf5_util.get_property(preprocessed_h5, file_structure.PreprocessedTensor2DJit.max_y)
        self._rasterizer = rasterizer.Rasterizer(self._scale, padding, min_x, max_x, min_y, max_y, square)
        self._transformer = transformer.Transformer(min_x, max_x, min_y, max_y)
        if hdf5_util.has_data_set(preprocessed_h5, file_structure.PreprocessedTensor2DJit.symbols):
            symbols = preprocessed_h5[file_structure.PreprocessedTensor2DJit.symbols]
            self._symbol_index_lookup = dict()
            for i in range(len(symbols)):
                self._symbol_index_lookup[symbols[i].decode('utf-8')] = i
            self._number_symbols = len(self._symbol_index_lookup)
        else:
            self._symbol_index_lookup = None
            self._number_symbols = 0
        if hdf5_util.has_data_set(preprocessed_h5, file_structure.PreprocessedTensor2DJit.chemical_properties):
            self._chemical_properties =\
                numpy_array_to_string_list(preprocessed_h5[file_structure.PreprocessedTensor2DJit.chemical_properties])
        else:
            self._chemical_properties = None
        self._gauss_sigma = hdf5_util.get_property(preprocessed_h5, file_structure.PreprocessedTensor2DJit.gauss_sigma)
        self._normalization_type = hdf5_util.get_property(preprocessed_h5,
                                                          file_structure.PreprocessedTensor2DJit.normalization_type)
        if hdf5_util.has_data_set(preprocessed_h5, file_structure.PreprocessedTensor2DJit.normalization_min):
            self._normalization_min = preprocessed_h5[file_structure.PreprocessedTensor2DJit.normalization_min][:]
        if hdf5_util.has_data_set(preprocessed_h5, file_structure.PreprocessedTensor2DJit.normalization_max):
            self._normalization_max = preprocessed_h5[file_structure.PreprocessedTensor2DJit.normalization_max][:]
        if hdf5_util.has_data_set(preprocessed_h5, file_structure.PreprocessedTensor2DJit.normalization_mean):
            self._normalization_mean = preprocessed_h5[file_structure.PreprocessedTensor2DJit.normalization_mean][:]
        if hdf5_util.has_data_set(preprocessed_h5, file_structure.PreprocessedTensor2DJit.normalization_std):
            self._normalization_std = preprocessed_h5[file_structure.PreprocessedTensor2DJit.normalization_std][:]
        preprocessed_h5.close()

    def preprocess(self, smiles_array, offset, queue, random_seed=None):
        for i in range(len(smiles_array)):
            if random_seed is not None:
                random_ = random.Random(random_seed + i)
            smiles = smiles_array[i].decode('utf-8')
            molecule = Chem.MolFromSmiles(smiles)
            AllChem.Compute2DCoords(molecule)
            successful = False
            while not successful:
                preprocessed_molecule = tensor_2d_jit_preprocessed.Tensor2DJitPreprocessed(i + offset)
                successful = True
                atom_positions = dict()
                if random_seed is not None:
                    rotation = random_.randint(0, 359)
                    flip = bool(random_.randint(0, 1))
                    shift_x = random_.randint(0, 1) / self._scale - 0.5 / self._scale
                    shift_y = random_.randint(0, 1) / self._scale - 0.5 / self._scale
                for atom in molecule.GetAtoms():
                    position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
                    position_x = position.x
                    position_y = position.y
                    if random_seed is not None:
                        position_x, position_y = self._transformer.apply(position_x, position_y, flip, rotation,
                                                                         shift_x, shift_y)
                    position_x, position_y = self._rasterizer.apply(position_x, position_y)
                    if position_x >= self.shape[0] or position_y >= self.shape[1]:
                        successful = False
                        break
                    symbol_index = None
                    if self._symbol_index_lookup is not None:
                        if atom.GetSymbol() in self._symbol_index_lookup:
                            symbol_index = self._symbol_index_lookup[atom.GetSymbol()]
                    chemical_property_values = None
                    if self._chemical_properties is not None:
                        chemical_property_values = chemical_properties.get_chemical_properties(atom, self._chemical_properties)
                        self.normalize(chemical_property_values)
                    preprocessed_molecule.add_atom(tensor_2d_jit_preprocessed.Tensor2DJitPreprocessedAtom(
                        position_x, position_y, symbol=symbol_index, features=chemical_property_values))
                    atom_positions[atom.GetIdx()] = [position_x, position_y]
            if self._with_bonds:
                bond_positions_ = bond_positions.calculate(molecule, atom_positions)
                for bond in molecule.GetBonds():
                    bond_symbol = bond_symbols.get_bond_symbol(bond.GetBondType())
                    if bond_symbol is not None and bond_symbol in self._symbol_index_lookup:
                        bond_symbol_index = self._symbol_index_lookup[bond_symbol]
                        for position in bond_positions_[bond.GetIdx()]:
                            preprocessed_molecule.add_atom(tensor_2d_jit_preprocessed.Tensor2DJitPreprocessedAtom(
                                position[0], position[1], symbol=bond_symbol_index))
            queue.put(preprocessed_molecule)
        if hasattr(queue, 'flush'):
            queue.flush()


    def substructure_locations(self, smiles_array, substructures, offset, locations_queue, random_seed=None,
                               only_substructures=False, only_atoms=False):
        for i in range(len(smiles_array)):
            if random_seed is not None:
                random_ = random.Random(random_seed + i)
            smiles = smiles_array[i].decode('utf-8')
            molecule = Chem.MolFromSmiles(smiles)
            AllChem.Compute2DCoords(molecule)
            indices = set()
            for substructure in substructures:
                matches = molecule.GetSubstructMatches(substructure)
                for match in matches:
                    for index in match:
                        indices.add(index)
            successful = False
            while not successful:
                successful = True
                atom_positions = dict()
                other_locations = list()
                substructure_locations_ = list()
                if random_seed is not None:
                    rotation = random_.randint(0, 359)
                    flip = bool(random_.randint(0, 1))
                    shift_x = random_.randint(0, 1) / self._scale - 0.5 / self._scale
                    shift_y = random_.randint(0, 1) / self._scale - 0.5 / self._scale
                for atom in molecule.GetAtoms():
                    position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
                    position_x = position.x
                    position_y = position.y
                    if random_seed is not None:
                        position_x, position_y = self._transformer.apply(position_x, position_y, flip, rotation,
                                                                         shift_x, shift_y)
                    position_x, position_y = self._rasterizer.apply(position_x, position_y)
                    if position_x >= self.shape[0] or position_y >= self.shape[1]:
                        successful = False
                        break
                    atom_positions[atom.GetIdx()] = [position_x, position_y]
                    if atom.GetIdx() in indices:
                        substructure_locations_.append([position_x, position_y])
                    else:
                        other_locations.append([position_x, position_y])
            if self._with_bonds and not only_atoms:
                bond_positions_ = bond_positions.calculate(molecule, atom_positions)
                for bond in molecule.GetBonds():
                    bond_symbol = bond_symbols.get_bond_symbol(bond.GetBondType())
                    if bond_symbol is not None and bond_symbol in self._symbol_index_lookup:
                        if bond.GetBeginAtomIdx() in indices and bond.GetEndAtomIdx() in indices:
                            for position in bond_positions_[bond.GetIdx()]:
                                substructure_locations_.append([position[0], position[1]])
                        else:
                            for position in bond_positions_[bond.GetIdx()]:
                                other_locations.append([position[0], position[1]])
            if only_substructures:
                locations_queue.put((i + offset, substructure_locations_))
            else:
                locations_queue.put((i + offset, substructure_locations_, other_locations))
        locations_queue.flush()


    @property
    def shape(self):
        return self._shape

    def normalize(self, values):
        if self._normalization_type == normalization.NormalizationTypes.min_max_1:
            values -= self._normalization_min
            values /= self._normalization_max - self._normalization_min
        if self._normalization_type == normalization.NormalizationTypes.min_max_2:
            values -= self._normalization_min
            values *= 2
            values /= self._normalization_max - self._normalization_min
            values -= 1
        if self._normalization_type == normalization.NormalizationTypes.z_score:
            values -= self._normalization_mean
            values /= self._normalization_std
        if numpy.inf in values or numpy.NINF in values:
            values[numpy.logical_or(values==numpy.inf, values==numpy.NINF)] = 0


def numpy_array_to_string_list(numpy_array):
    string_list = list()
    for i in range(len(numpy_array)):
        string_list.append(numpy_array[i].decode('utf-8'))
    return string_list
