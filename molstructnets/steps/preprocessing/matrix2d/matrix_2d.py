from util import data_validation, misc, file_structure, file_util, logger, progressbar, concurrent_max, concurrent_set,\
    thread_pool, constants, hdf5_util, concurrent_min, reference_data_set
from steps.preprocessing.matrix2d import bond_positions
import h5py
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
import numpy
import sys


number_threads = thread_pool.default_number_threads
bond_symbols = {'-', '=', '#', '$', ':'}


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
        parameters.append({'id': 'scale', 'name': 'Scale factor (default: 2)', 'type': float, 'default': 2.0,
                           'description': 'The scaling factor used to change the size of the resulting grid.'})
        parameters.append({'id': 'width', 'name': 'Width (default: automatic)', 'type': int, 'default': None,
                           'description': 'The maximum width of molecules that will fit into the preprocessed data'
                                          ' array.'})
        parameters.append({'id': 'height', 'name': 'Height (default: automatic)', 'type': int, 'default': None,
                           'description': 'The maximum height of molecules that will fit into the preprocessed data'
                                          ' array.'})
        parameters.append({'id': 'symbols', 'name': 'Force symbols (default: none)', 'type': str, 'default': None,
                           'description': 'Symbols in the given string will be added to the index in addition to'
                                          ' symbols found in the data set. Multiple symbols are separated by a'
                                          ' semicolon (e.g. \'F;Cl;S\').'})
        parameters.append({'id': 'transformations', 'name': 'Number of transformations for each matrix (default: none)',
                           'type': int, 'default': 0,
                           'description': 'The number of transformations done for each matrix in the training data'
                                          ' set.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(global_parameters, [constants.GlobalParameters.seed])
        hash_parameters.update(misc.copy_dict_from_keys(local_parameters, ['scale', 'width', 'height', 'symbols',
                                                                           'transformations']))
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
                        pool.submit(Matrix2D.analyze_smiles, smiles_data[chunk['start']:chunk['end'] + 1], symbols,
                                    max_nr_atoms, min_x, min_y, max_x, max_y, local_parameters['scale'], progress)
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
            index = hdf5_util.create_dataset(preprocessed_h5, 'index', (len(symbols),),
                                             dtype='S' + str(max_symbol_length))
            index_lookup = {}
            for i in range(len(symbols)):
                index_lookup[symbols[i]] = i
                index[i] = symbols[i].encode('utf-8')
            global_parameters[constants.GlobalParameters.input_dimensions] = (max_x + 1, max_y + 1, len(index))
            preprocessed = hdf5_util.create_dataset(preprocessed_h5, file_structure.Preprocessed.preprocessed,
                                                    (len(smiles_data), max_x + 1, max_y + 1, len(index)), dtype='I',
                                                    chunks=(1, max_x + 1, max_y + 1, len(index)))
            atom_locations = hdf5_util.create_dataset(preprocessed_h5, 'atom_locations',
                                                      (len(smiles_data), max_nr_atoms, 2), dtype='I',
                                                      chunks=(1, max_nr_atoms, 2))
            logger.log('Writing matrices')
            with progressbar.ProgressBar(len(smiles_data)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(Matrix2D.write_2d_matrices, preprocessed, atom_locations,
                                    smiles_data[chunk['start']:chunk['end'] + 1], index_lookup, min_x, min_y,
                                    chunk['start'], local_parameters['scale'], progress)
                    pool.wait()
            if local_parameters['transformations'] > 0:
                logger.log('Writing transformed training data')
                partition_h5 = h5py.File(global_parameters[constants.GlobalParameters.partition_data], 'r')
                train = partition_h5[file_structure.Partitions.train]
                preprocessed_training = \
                    hdf5_util.create_dataset(preprocessed_h5, file_structure.Preprocessed.preprocessed_training,
                                             (train.shape[0] * local_parameters['transformations'],
                                              preprocessed.shape[1], preprocessed.shape[2], preprocessed.shape[3]),
                                             dtype='I', chunks=(1, preprocessed.shape[1], preprocessed.shape[2],
                                                                preprocessed.shape[3]))
                preprocessed_training_ref = \
                    hdf5_util.create_dataset(preprocessed_h5,
                                             file_structure.Preprocessed.preprocessed_training_references,
                                             (preprocessed_training.shape[0],), dtype='I')
                train_smiles_data = reference_data_set.ReferenceDataSet(train, smiles_data)
                # If we do the transformation in parallel it leads to race conditions for the original data point
                # (oversampled data). In this case the same seed does not lead to the same results. For this reason
                # parallelization is disabled
                chunks = misc.chunk(len(train), 1)
                originals_set = concurrent_set.ConcurrentSet()
                with progressbar.ProgressBar(len(preprocessed_training)) as progress:
                    with thread_pool.ThreadPool(len(chunks)) as pool:
                        for chunk in chunks:
                            pool.submit(Matrix2D.write_transformed_2d_matrices, preprocessed_training,
                                        preprocessed_training_ref, train_smiles_data[chunk['start']:chunk['end'] + 1],
                                        index_lookup, min_x, min_y, chunk['start'], local_parameters['scale'],
                                        originals_set, local_parameters['transformations'], len(train_smiles_data),
                                        global_parameters[constants.GlobalParameters.seed] + chunk['start'], train,
                                        progress)
                        pool.wait()
            data_h5.close()
            preprocessed_h5.close()
            file_util.move_file(temp_preprocessed_path, preprocessed_path)

    @staticmethod
    def analyze_smiles(smiles_data, symbols, max_nr_atoms, min_x, min_y, max_x, max_y, scale_factor, progress):
        soft_limit = 100 * scale_factor
        local_min_x = None
        local_max_x = None
        local_min_y = None
        local_max_y = None
        for smiles in smiles_data:
            smiles = smiles.decode('utf-8')
            molecule = Chem.MolFromSmiles(smiles)
            AllChem.Compute2DCoords(molecule)
            max_nr_atoms.add_value(len(molecule.GetAtoms()))
            for atom in molecule.GetAtoms():
                symbols.add(atom.GetSymbol())
                position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
                x = round(position.x * scale_factor)
                y = round(position.y * scale_factor)
                local_min_x = misc.minimum(local_min_x, x)
                local_max_x = misc.maximum(local_max_x, x)
                local_min_y = misc.minimum(local_min_y, y)
                local_max_y = misc.maximum(local_max_y, y)
            min_x.add_value(local_min_x)
            max_x.add_value(local_max_x)
            min_y.add_value(local_min_y)
            max_y.add_value(local_max_y)
            # Warn about very big molecules
            size_x = local_max_x - local_min_x
            size_y = local_max_y - local_min_y
            if size_x > soft_limit or size_y > soft_limit:
                logger.log('Encountered big molecule with size ' + str(size_x) + '×' + str(size_y) + ': '
                           + smiles, logger.LogLevel.WARNING)
            progress.increment()

    @staticmethod
    def write_2d_matrices(preprocessed, atom_locations, smiles_data, index_lookup, min_x, min_y, offset, scale_factor,
                          progress):
        for i in range(len(smiles_data)):
            smiles = smiles_data[i].decode('utf-8')
            molecule = Chem.MolFromSmiles(smiles)
            AllChem.Compute2DCoords(molecule)
            preprocessed_row, atom_locations_row = Matrix2D.create_2d_matrix(molecule, index_lookup, scale_factor,
                                                                             min_x, min_y, preprocessed.shape,
                                                                             atom_locations_shape=atom_locations.shape)
            preprocessed[i + offset, :] = preprocessed_row[:]
            atom_locations[i + offset, :] = atom_locations_row[:]
            progress.increment()

    @staticmethod
    def write_transformed_2d_matrices(preprocessed_training, preprocessed_training_ref, smiles_data, index_lookup,
                                      min_x, min_y, offset, scale_factor, originals_set, number_transformations,
                                      offset_per_transformation, seed, train_ref, progress):
        random_ = numpy.random.RandomState(seed)
        for i in range(len(smiles_data)):
            original_index = train_ref[i + offset]
            smiles = smiles_data[i].decode('utf-8')
            molecule = Chem.MolFromSmiles(smiles)
            number_atoms = len(molecule.GetAtoms())
            number_distances = int((number_atoms * number_atoms - number_atoms) / 2)
            start = 0
            if originals_set.add(smiles):
                AllChem.Compute2DCoords(molecule)
                preprocessed_row = Matrix2D.create_2d_matrix(molecule, index_lookup, scale_factor, min_x, min_y,
                                                             preprocessed_training.shape)[0]
                preprocessed_training[i + offset, :] = preprocessed_row[:]
                preprocessed_training_ref[i + offset] = original_index
                start += 1
                progress.increment()
            failed_attempts = 0
            for j in range(start, number_transformations):
                invalid = True
                while invalid:
                    if failed_attempts < 10:
                        distances = random_.rand(number_distances)
                        AllChem.Compute2DCoordsMimicDistmat(molecule, distances)
                    else:
                        AllChem.Compute2DCoords(molecule)
                    invalid = not Matrix2D.fits(molecule, min_x, min_y, preprocessed_training.shape[1] - 1,
                                                preprocessed_training.shape[2] - 1, scale_factor)
                    if invalid:
                        failed_attempts += 1
                index = i + offset + offset_per_transformation * j
                preprocessed_row = Matrix2D.create_2d_matrix(molecule, index_lookup, scale_factor, min_x, min_y,
                                                             preprocessed_training.shape)[0]
                preprocessed_training[index, :] = preprocessed_row[:]
                preprocessed_training_ref[index] = original_index
                progress.increment()

    @staticmethod
    def fits(molecule, min_x, min_y, max_x, max_y, scale_factor):
        invalid = False
        for atom in molecule.GetAtoms():
            position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
            x = round(position.x * scale_factor) - min_x
            y = round(position.y * scale_factor) - min_y
            if not (0 <= x <= max_x and 0 <= y <= max_y):
                invalid = True
                break
        return not invalid

    @staticmethod
    def create_2d_matrix(molecule, index_lookup, scale_factor, min_x, min_y, preprocessed_shape,
                         atom_locations_shape=None):
        preprocessed_row = numpy.zeros((preprocessed_shape[1], preprocessed_shape[2], preprocessed_shape[3]),
                                       dtype='int16')
        if atom_locations_shape is None:
            atom_locations_row = None
        else:
            atom_locations_row = numpy.zeros((atom_locations_shape[1], atom_locations_shape[2]), dtype='int16')
        atom_positions = dict()
        for atom in molecule.GetAtoms():
            symbol_index = index_lookup[atom.GetSymbol()]
            position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
            x = round(position.x * scale_factor) - min_x
            y = round(position.y * scale_factor) - min_y
            preprocessed_row[x, y, symbol_index] = 1
            if atom_locations_row is not None:
                atom_locations_row[atom.GetIdx(), 0] = x
                atom_locations_row[atom.GetIdx(), 1] = y
            atom_positions[atom.GetIdx()] = [x, y]
        bond_positions_ = bond_positions.calculate(molecule, atom_positions)
        for bond in molecule.GetBonds():
            bond_symbol = Matrix2D.get_bond_symbol(bond.GetBondType())
            if bond_symbol is not None:
                bond_symbol_index = index_lookup[bond_symbol]
                for position in bond_positions_[bond.GetIdx()]:
                    preprocessed_row[position[0], position[1], bond_symbol_index] = 1
        return preprocessed_row, atom_locations_row

    @staticmethod
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
