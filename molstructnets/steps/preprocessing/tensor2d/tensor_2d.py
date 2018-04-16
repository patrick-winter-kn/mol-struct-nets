import h5py
from rdkit import Chem
from rdkit.Chem import AllChem

from steps.preprocessing.shared.tensor2d import rasterizer, molecule_2d_tensor
from util import data_validation, misc, file_structure, file_util, logger, progressbar, concurrent_max,\
    concurrent_set, thread_pool, constants, hdf5_util, concurrent_min, normalization
from steps.preprocessing.shared.chemicalproperties import chemical_properties


number_threads = thread_pool.default_number_threads
fixed_symbols = {'-', '=', '#', '$', ':'}
if molecule_2d_tensor.with_empty_bits:
    fixed_symbols.add(' ')


class Tensor2D:

    @staticmethod
    def get_id():
        return 'tensor_2d'

    @staticmethod
    def get_name():
        return '2D Tensor'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'scale', 'name': 'Scale Factor', 'type': float, 'default': 2.0, 'min': 0.1,
                           'description': 'The scaling factor used to change the size of the resulting grid. Default:'
                                          ' 2.0'})
        parameters.append({'id': 'symbols', 'name': 'Force Symbols', 'type': str, 'default': None,
                           'description': 'Symbols in the given string will be added to the index in addition to'
                                          ' symbols found in the data set. Multiple symbols are separated by a'
                                          ' semicolon (e.g. \'F;Cl;S\'). Default: None'})
        parameters.append({'id': 'square', 'name': 'Same Size for X and Y Dimensions', 'type': bool, 'default': True,
                           'description': 'Make the size of the X and Y dimensions the same. This is important if the'
                                          ' training data should be transformed (in order to fit rotated data).'
                                          ' Default: True'})
        parameters.append({'id': 'bonds', 'name': 'With bonds', 'type': bool, 'default': True,
                           'description': 'Add symbols for the bonds. Default: True'})
        parameters.append({'id': 'chemical_properties', 'name': 'With chemical properties', 'type': bool,
                           'default': False, 'description': 'Adds chemical properties to the data. Default: False'})
        parameters.append({'id': 'normalize', 'name': 'Normalize values', 'type': bool, 'default': False,
                           'description': 'Normalize input values. Default: False'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(local_parameters,
                                                   ['scale', 'symbols', 'square', 'bonds', 'chemical_properties',
                                                    'normalize'])
        file_name = 'tensor_2d_' + misc.hash_parameters(hash_parameters) + '.h5'
        return file_util.resolve_subpath(file_structure.get_preprocessed_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        preprocessed_path = Tensor2D.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.preprocessed_data] = preprocessed_path
        file_exists = file_util.file_exists(preprocessed_path)
        needs_normalization = local_parameters['normalize']
        if file_exists:
            preprocessed_h5 = h5py.File(preprocessed_path, 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            global_parameters[constants.GlobalParameters.input_dimensions] = (preprocessed.shape[1],
                                                                              preprocessed.shape[2],
                                                                              preprocessed.shape[3])
            if file_structure.Preprocessed.preprocessed_normalization_stats in preprocessed_h5:
                needs_normalization = False
            preprocessed_h5.close()
        if file_exists and not needs_normalization:
            logger.log('Skipping step: ' + preprocessed_path + ' already exists')
        else:
            if not file_exists:
                data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
                smiles_data = data_h5[file_structure.DataSet.smiles]
                temp_preprocessed_path = file_util.get_temporary_file_path('tensor_2d')
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
                            pool.submit(Tensor2D.analyze_smiles, smiles_data[chunk['start']:chunk['end'] + 1], symbols,
                                        max_nr_atoms, min_x, min_y, max_x, max_y, progress)
                        pool.wait()
                max_nr_atoms = max_nr_atoms.get_max()
                min_x = min_x.get_min()
                min_y = min_y.get_min()
                max_x = max_x.get_max()
                max_y = max_y.get_max()
                new_symbols = set()
                if local_parameters['bonds']:
                    new_symbols |= fixed_symbols
                if not local_parameters['chemical_properties']:
                    new_symbols |= symbols.get_set_copy()
                symbols = sorted(new_symbols)
                max_symbol_length = 1
                for symbol in symbols:
                    max_symbol_length = max(max_symbol_length, len(symbol))
                index = hdf5_util.create_dataset(preprocessed_h5, 'index', (len(symbols),),
                                                 dtype='S' + str(max_symbol_length))
                index_lookup = {}
                for i in range(len(symbols)):
                    index_lookup[symbols[i]] = i
                    index[i] = symbols[i].encode('utf-8')
                rasterizer_ = rasterizer.Rasterizer(local_parameters['scale'], molecule_2d_tensor.padding, min_x, max_x,
                                                    min_y, max_y, local_parameters['square'])
                feature_vector_size = len(index)
                if local_parameters['chemical_properties']:
                    feature_vector_size += len(chemical_properties.Properties.selected)
                global_parameters[constants.GlobalParameters.input_dimensions] = (rasterizer_.size_x,
                                                                                  rasterizer_.size_y,
                                                                                  feature_vector_size)
                preprocessed = hdf5_util.create_dataset(preprocessed_h5, file_structure.Preprocessed.preprocessed,
                                                        (len(smiles_data), rasterizer_.size_x, rasterizer_.size_y,
                                                         feature_vector_size), dtype='f',
                                                        chunks=(1, rasterizer_.size_x, rasterizer_.size_y,
                                                                feature_vector_size))
                atom_locations = hdf5_util.create_dataset(preprocessed_h5, 'atom_locations',
                                                          (len(smiles_data), max_nr_atoms, 2), dtype='int16',
                                                          chunks=(1, max_nr_atoms, 2))
                logger.log('Writing tensors')
                with progressbar.ProgressBar(len(smiles_data)) as progress:
                    with thread_pool.ThreadPool(number_threads) as pool:
                        for chunk in chunks:
                            pool.submit(Tensor2D.write_2d_tensors, preprocessed, atom_locations,
                                        smiles_data[chunk['start']:chunk['end'] + 1], index_lookup, rasterizer_,
                                        chunk['start'], local_parameters['chemical_properties'], progress)
                        pool.wait()
                data_h5.close()
                preprocessed_h5.close()
                hdf5_util.set_property(temp_preprocessed_path, 'scale', local_parameters['scale'])
                hdf5_util.set_property(temp_preprocessed_path, 'min_x', min_x)
                hdf5_util.set_property(temp_preprocessed_path, 'max_x', max_x)
                hdf5_util.set_property(temp_preprocessed_path, 'min_y', min_y)
                hdf5_util.set_property(temp_preprocessed_path, 'max_y', max_y)
                hdf5_util.set_property(temp_preprocessed_path, 'chemical_properties',
                                       local_parameters['chemical_properties'])
                file_util.move_file(temp_preprocessed_path, preprocessed_path)
            if needs_normalization:
                normalization.normalize_data_set(preprocessed_path, file_structure.Preprocessed.preprocessed)


    @staticmethod
    def analyze_smiles(smiles_data, symbols, max_nr_atoms, min_x, min_y, max_x, max_y, progress):
        soft_limit = 100
        local_min_x = None
        local_max_x = None
        local_min_y = None
        local_max_y = None
        for smiles in smiles_data:
            smiles = smiles.decode('utf-8')
            molecule = Chem.MolFromSmiles(smiles)
            AllChem.Compute2DCoords(molecule)
            max_nr_atoms.add(len(molecule.GetAtoms()))
            for atom in molecule.GetAtoms():
                symbols.add(atom.GetSymbol())
                position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
                x = position.x
                y = position.y
                local_min_x = misc.minimum(local_min_x, x)
                local_max_x = misc.maximum(local_max_x, x)
                local_min_y = misc.minimum(local_min_y, y)
                local_max_y = misc.maximum(local_max_y, y)
            min_x.add(local_min_x)
            max_x.add(local_max_x)
            min_y.add(local_min_y)
            max_y.add(local_max_y)
            # Warn about very big molecules
            size_x = local_max_x - local_min_x
            size_y = local_max_y - local_min_y
            if size_x > soft_limit or size_y > soft_limit:
                logger.log('Encountered big molecule with size ' + str(size_x) + '×' + str(size_y) + ': '
                           + smiles, logger.LogLevel.WARNING)
            if progress is not None:
                progress.increment()

    @staticmethod
    def write_2d_tensors(preprocessed, atom_locations, smiles_data, index_lookup, rasterizer_, offset,
                         with_chemical_properties, progress):
        for i in range(len(smiles_data)):
            smiles = smiles_data[i].decode('utf-8')
            molecule = Chem.MolFromSmiles(smiles)
            preprocessed_row, atom_locations_row =\
                molecule_2d_tensor.molecule_to_2d_tensor(molecule, index_lookup, rasterizer_, preprocessed.shape,
                                                         atom_locations_shape=atom_locations.shape,
                                                         with_chemical_properties=with_chemical_properties)
            preprocessed[i + offset, :] = preprocessed_row[:]
            atom_locations[i + offset, :] = atom_locations_row[:]
            progress.increment()
