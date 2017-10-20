from util import data_validation, misc, file_structure, file_util, logger, progressbar, concurrent_max, concurrent_set,\
    thread_pool, constants, hdf5_util, concurrent_min
import h5py
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
import numpy


number_threads = thread_pool.default_number_threads
bond_symbols = {'-', '=', '#', '$', ':'}
layout_factor = 2


class Matrix2D:

    @staticmethod
    def get_id():
        return 'matrix_2d'

    @staticmethod
    def get_name():
        return 'Matrix 2D'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'width', 'name': 'Width (default: automatic)', 'type': int,
                           'default': None,
                           'description': 'The maximum width of molecules that will fit into the preprocessed data'
                                          ' array.'})
        parameters.append({'id': 'height', 'name': 'Height (default: automatic)', 'type': int,
                           'default': None,
                           'description': 'The maximum height of molecules that will fit into the preprocessed data'
                                          ' array.'})
        parameters.append({'id': 'symbols', 'name': 'Force symbols (default: none)', 'type': str,
                           'default': None, 'description': 'Symbols in the given string will be added to the index'
                                                           ' in addition to symbols found in the data set. Multiple'
                                                           ' symbols are separated by a semicolon (e.g. \'F;Cl;S\').'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(local_parameters, ['width', 'height', 'symbols'])
        file_name = 'matrix_2d_' + misc.hash_parameters(hash_parameters) + '.h5'
        return file_util.resolve_subpath(file_structure.get_preprocessed_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        preprocessed_path = Matrix2D.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.preprocessed_data] = preprocessed_path
        if file_util.file_exists(preprocessed_path):
            logger.log('Skipping step: ' + preprocessed_path + ' already exists')
            preprocessed_h5 = h5py.File(preprocessed_path, 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            global_parameters[constants.GlobalParameters.input_dimensions] = (preprocessed.shape[1],
                                                                              preprocessed.shape[2],
                                                                              preprocessed.shape[3])
            preprocessed_h5.close()
        else:
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles_data = data_h5[file_structure.DataSet.smiles]
            temp_preprocessed_path = file_util.get_temporary_file_path('matrix_2d')
            preprocessed_h5 = h5py.File(temp_preprocessed_path, 'w')
            chunks = misc.chunk(len(smiles_data), number_threads)
            symbols = concurrent_set.ConcurrentSet()
            if local_parameters['symbols'] is not None:
                for symbol in local_parameters['symbols'].split(';'):
                    symbols.add(symbol)
            max_nr_atoms = concurrent_max.ConcurrentMax()
            min_x = concurrent_min.ConcurrentMin()
            min_y = concurrent_min.ConcurrentMin()
            max_x = concurrent_max.ConcurrentMax()
            max_y = concurrent_max.ConcurrentMax()
            logger.log('Analyzing SMILES')
            with progressbar.ProgressBar(len(smiles_data)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(Matrix2D.analyze_smiles, smiles_data[chunk['start']:chunk['end'] + 1],
                                    symbols, max_nr_atoms, min_x, min_y, max_x, max_y, progress)
                    pool.wait()
            max_nr_atoms = max_nr_atoms.get_max()
            min_x = min_x.get_min()
            min_y = min_y.get_min()
            max_x = max_x.get_max() - min_x
            max_y = max_y.get_max() - min_y
            if local_parameters['width'] is not None:
                max_x = max(max_x, local_parameters['width'])
            if local_parameters['height'] is not None:
                max_y = max(max_y, local_parameters['height'])
            symbols = sorted(symbols.get_set_copy() | bond_symbols)
            max_symbol_length = 0
            for symbol in symbols:
                max_symbol_length = max(max_symbol_length, len(symbol))
            index = hdf5_util.create_dataset(preprocessed_h5, 'index', (len(symbols),), dtype='S' + str(max_symbol_length))
            index_lookup = {}
            for i in range(len(symbols)):
                index_lookup[symbols[i]] = i
                index[i] = symbols[i].encode('utf-8')
            global_parameters[constants.GlobalParameters.input_dimensions] = (max_x + 1, max_y + 1, len(index))
            preprocessed = hdf5_util.create_dataset(preprocessed_h5, file_structure.Preprocessed.preprocessed,
                                                          (len(smiles_data), max_x + 1, max_y + 1, len(index)),
                                                          dtype='I', chunks=(1, max_x + 1, max_y + 1, len(index)))
            atom_locations = hdf5_util.create_dataset(preprocessed_h5, 'atom_locations',
                                                          (len(smiles_data), max_nr_atoms, 2),
                                                          dtype='I', chunks=(1, max_nr_atoms, 2))
            logger.log('Writing matrices')
            with progressbar.ProgressBar(len(smiles_data)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(Matrix2D.write_2d_matrices, preprocessed, atom_locations,
                                    smiles_data[chunk['start']:chunk['end'] + 1], index_lookup, min_x, min_y,
                                    chunk['start'], progress)
                    pool.wait()
            data_h5.close()
            preprocessed_h5.close()
            file_util.move_file(temp_preprocessed_path, preprocessed_path)

    @staticmethod
    def analyze_smiles(smiles_data, symbols, max_nr_atoms, min_x, min_y, max_x, max_y, progress):
        for smiles in smiles_data:
            smiles = smiles.decode('utf-8')
            molecule = Chem.MolFromSmiles(smiles)
            AllChem.Compute2DCoords(molecule)
            max_nr_atoms.add_value(len(molecule.GetAtoms()))
            for atom in molecule.GetAtoms():
                symbols.add(atom.GetSymbol())
                position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
                x = round(position.x * layout_factor)
                y = round(position.y * layout_factor)
                min_x.add_value(x)
                max_x.add_value(x)
                min_y.add_value(y)
                max_y.add_value(y)
            progress.increment()

    @staticmethod
    def write_2d_matrices(preprocessed, atom_locations, smiles_data, index_lookup, min_x, min_y, offset, progress):
        for i in range(len(smiles_data)):
            preprocessed_row = numpy.zeros((preprocessed.shape[1], preprocessed.shape[2], preprocessed.shape[3]), dtype='int16')
            atom_locations_row = numpy.zeros((atom_locations.shape[1], atom_locations.shape[2]), dtype='int16')
            smiles = smiles_data[i].decode('utf-8')
            molecule = Chem.MolFromSmiles(smiles)
            AllChem.Compute2DCoords(molecule)
            atom_positions = dict()
            for atom in molecule.GetAtoms():
                symbol_index = index_lookup[atom.GetSymbol()]
                position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
                x = round(position.x * layout_factor) - min_x
                y = round(position.y * layout_factor) - min_y
                preprocessed_row[x, y, symbol_index] = 1
                atom_locations_row[atom.GetIdx(), 0] = x
                atom_locations_row[atom.GetIdx(), 1] = y
                atom_positions[atom.GetIdx()] = [x, y]
            for bond in molecule.GetBonds():
                bond_symbol = get_bond_symbol(bond.GetBondType())
                if bond_symbol is not None:
                    x1 = atom_positions[bond.GetBeginAtomIdx()][0]
                    y1 = atom_positions[bond.GetBeginAtomIdx()][1]
                    x2 = atom_positions[bond.GetEndAtomIdx()][0]
                    y2 = atom_positions[bond.GetEndAtomIdx()][1]
                    f = LinearFunction(x1, y1, x2, y2)
                    bond_symbol_index = index_lookup[bond_symbol]
                    for x in range(min(x1, x2) + 1, max(x1, x2)):
                        y = round(f.get_y(x))
                        if preprocessed_row[x, y].max() == 0:
                            preprocessed_row[x, y, bond_symbol_index] = 1
                    for y in range(min(y1, y2) + 1, max(y1, y2)):
                        x = round(f.get_x(y))
                        if preprocessed_row[x, y].max() == 0:
                            preprocessed_row[x, y, bond_symbol_index] = 1
            preprocessed[i + offset, :] = preprocessed_row[:]
            atom_locations[i + offset, :] = atom_locations_row[:]
            progress.increment()


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
