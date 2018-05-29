import h5py
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy
from steps.preprocessing.shared.tensor2d import molecule_2d_tensor, bond_symbols, rasterizer
from util import data_validation, misc, file_structure, file_util, logger, thread_pool, hdf5_util, normalization,\
    process_pool, constants
from steps.preprocessing.shared.chemicalproperties import chemical_properties


number_threads = thread_pool.default_number_threads
fixed_symbols = {'-', '=', '#', '$', ':'}
if molecule_2d_tensor.with_empty_bits:
    fixed_symbols.add(' ')


class Tensor2DJit:

    @staticmethod
    def get_id():
        return 'tensor_2d_jit'

    @staticmethod
    def get_name():
        return '2D Tensor JIT'

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
        parameters.append({'id': 'bonds', 'name': 'With Bonds', 'type': bool, 'default': True,
                           'description': 'Add symbols for the bonds. Default: True'})
        parameters.append({'id': 'atom_symbols', 'name': 'With Atom Symbols', 'type': bool,
                           'default': True, 'description': 'Adds atom symbols to the data. Default: True'})
        parameters.append({'id': 'chemical_properties', 'name': 'Chemical Properties', 'type': list,
                           'default': [], 'options': chemical_properties.Properties.all,
                           'description': 'The chemical properties that will be used. Default: None'})
        parameters.append({'id': 'gauss_sigma', 'name': 'Gauss sigma', 'type': float, 'default': None, 'min': 0,
                           'description': 'Sigma for gauss filter. Default: No gauss filter'})
        parameters.append({'id': 'normalization', 'name': 'Normalization Type', 'type': str, 'default': None,
                           'options': ['None',
                                       normalization.NormalizationTypes.min_max_1,
                                       normalization.NormalizationTypes.min_max_2,
                                       normalization.NormalizationTypes.z_score],
                           'description': 'Normalization type. Default: None'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(local_parameters,
                                                   ['scale', 'symbols', 'square', 'bonds', 'chemical_properties',
                                                    'gauss_sigma', 'normalization'])
        file_name = 'tensor_2d_jit_' + misc.hash_parameters(hash_parameters) + '.h5'
        return file_util.resolve_subpath(file_structure.get_preprocessed_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        preprocessed_path = Tensor2DJit.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.preprocessed_data] = preprocessed_path
        if file_util.file_exists(preprocessed_path):
            global_parameters[constants.GlobalParameters.input_dimensions] =\
                tuple(hdf5_util.get_property(preprocessed_path, file_structure.PreprocessedTensor2DJit.dimensions))
            logger.log('Skipping step: ' + preprocessed_path + ' already exists')
        else:
            temp_preprocessed_path = file_util.get_temporary_file_path('tensor_2d')
            preprocessed_h5 = h5py.File(temp_preprocessed_path, 'w')
            number_chemical_properties = len(local_parameters['chemical_properties'])
            # Write chemical properties
            if number_chemical_properties > 0:
                chemical_properties_array = Tensor2DJit.string_list_to_numpy_array(
                    local_parameters['chemical_properties'])
                hdf5_util.create_dataset_from_data(preprocessed_h5,
                                                   file_structure.PreprocessedTensor2DJit.chemical_properties,
                                                   chemical_properties_array)
            # First run: Calculate gridsize_x, gridsize_y, symbols, normalization_min, normalization_max,
            # normalization_mean
            data_set_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles = data_set_h5[file_structure.DataSet.smiles][:]
            chunks = misc.chunk(smiles.shape[0], process_pool.default_number_processes)
            logger.log('Calculating stats')
            with process_pool.ProcessPool(len(chunks)) as pool:
                needs_min_max = local_parameters['normalization'] == normalization.NormalizationTypes.min_max_1\
                                or local_parameters['normalization'] == normalization.NormalizationTypes.min_max_2
                needs_mean_std = local_parameters['normalization'] == normalization.NormalizationTypes.z_score
                for chunk in chunks:
                    pool.submit(Tensor2DJit.first_run, smiles[chunk['start']:chunk['end'] + 1],
                                chemical_properties_=local_parameters['chemical_properties'],
                                with_atom_symbols=local_parameters['atom_symbols'],
                                with_bonds=local_parameters['bonds'],
                                normalization_min=needs_min_max, normalization_max=needs_min_max,
                                normalization_mean=needs_mean_std)
                results = pool.get_results()
                min_x = None
                max_x = None
                min_y = None
                max_y = None
                symbols = set()
                if needs_min_max:
                    mins = numpy.zeros(number_chemical_properties, dtype='float32')
                    mins[:] = numpy.nan
                    maxs = numpy.zeros(number_chemical_properties, dtype='float32')
                    maxs[:] = numpy.nan
                if needs_mean_std:
                    means = numpy.zeros(number_chemical_properties, dtype='float32')
                for i in range(len(results)):
                    min_x = misc.minimum(min_x, results[i]['min_x'])
                    max_x = misc.minimum(max_x, results[i]['max_x'])
                    min_y = misc.minimum(min_y, results[i]['min_y'])
                    max_y = misc.minimum(max_y, results[i]['max_y'])
                    symbols = symbols.union(results[i]['symbols'])
                    if needs_min_max or needs_mean_std:
                        for j in range(number_chemical_properties):
                            if needs_min_max:
                                mins[j] = misc.minimum(mins[j], results[i]['normalization_min'][j])
                                maxs[j] = misc.maximum(maxs[j], results[i]['normalization_max'][j])
                            if needs_mean_std:
                                means[j] += results[i]['normalization_mean'][j]
                if local_parameters['symbols'] is not None:
                    symbols = symbols.union(set(local_parameters['symbols'].split(';')))
                if needs_mean_std:
                    means /= len(chunks)
                if len(symbols) > 0:
                    symbols = Tensor2DJit.string_list_to_numpy_array(sorted(symbols))
                    hdf5_util.create_dataset_from_data(preprocessed_h5, file_structure.PreprocessedTensor2DJit.symbols,
                                                       symbols)
                if needs_min_max:
                    hdf5_util.create_dataset_from_data(preprocessed_h5,
                                                       file_structure.PreprocessedTensor2DJit.normalization_min, mins)
                    hdf5_util.create_dataset_from_data(preprocessed_h5,
                                                       file_structure.PreprocessedTensor2DJit.normalization_max, maxs)
                if needs_mean_std:
                    hdf5_util.create_dataset_from_data(preprocessed_h5,
                                                       file_structure.PreprocessedTensor2DJit.normalization_mean, means)
                rasterizer_ = rasterizer.Rasterizer(local_parameters['scale'], molecule_2d_tensor.padding, min_x, max_x,
                                                    min_y, max_y, local_parameters['square'])
                dimensions = (rasterizer_.size_x, rasterizer_.size_y, len(symbols) +
                              len(local_parameters['chemical_properties']))
                # Second run: Calculate normalization_std
                if needs_mean_std:
                    logger.log('Calculating standard deviation')
                    for chunk in chunks:
                        pool.submit(Tensor2DJit.second_run, smiles[chunk['start']:chunk['end'] + 1],
                                    local_parameters['chemical_properties'], means)
                    results = pool.get_results()
                    stds = numpy.zeros(number_chemical_properties, dtype='float32')
                    atom_counter = 0
                    for i in range(len(results)):
                        atom_counter += results[i]['atom_counter']
                        for j in range(number_chemical_properties):
                            stds[j] += results[i]['normalization_std'][j]
                    stds /= atom_counter
                    stds = numpy.sqrt(stds)
                    hdf5_util.create_dataset_from_data(preprocessed_h5,
                                                       file_structure.PreprocessedTensor2DJit.normalization_std, stds)
            preprocessed_h5.close()
            data_set_h5.close()
            hdf5_util.set_property(temp_preprocessed_path, file_structure.PreprocessedTensor2DJit.dimensions,
                                   dimensions)
            hdf5_util.set_property(temp_preprocessed_path, file_structure.PreprocessedTensor2DJit.min_x, min_x)
            hdf5_util.set_property(temp_preprocessed_path, file_structure.PreprocessedTensor2DJit.max_x, max_x)
            hdf5_util.set_property(temp_preprocessed_path, file_structure.PreprocessedTensor2DJit.min_y, min_y)
            hdf5_util.set_property(temp_preprocessed_path, file_structure.PreprocessedTensor2DJit.max_y, max_y)
            hdf5_util.set_property(temp_preprocessed_path, file_structure.PreprocessedTensor2DJit.scale,
                                   local_parameters['scale'])
            hdf5_util.set_property(temp_preprocessed_path, file_structure.PreprocessedTensor2DJit.square,
                                   local_parameters['square'])
            if local_parameters['gauss_sigma'] is not None:
                hdf5_util.set_property(temp_preprocessed_path, file_structure.PreprocessedTensor2DJit.gauss_sigma,
                                       local_parameters['gauss_sigma'])
            if local_parameters['normalization'] is not None:
                hdf5_util.set_property(temp_preprocessed_path, file_structure.PreprocessedTensor2DJit.normalization_type,
                                       local_parameters['normalization'])
            file_util.move_file(temp_preprocessed_path, preprocessed_path)
            global_parameters[constants.GlobalParameters.input_dimensions] = dimensions

    @staticmethod
    def first_run(smiles, chemical_properties_=[], with_atom_symbols=False, with_bonds=False, normalization_min=False,
                  normalization_max=False, normalization_mean=False):
        results = dict()
        min_x = None
        max_x = None
        min_y = None
        max_y = None
        symbols = set()
        if normalization_min:
            n_min = numpy.zeros(len(chemical_properties_), dtype='float32')
            n_min[:] = numpy.nan
        if normalization_max:
            n_max = numpy.zeros(len(chemical_properties_), dtype='float32')
            n_max[:] = numpy.nan
        if normalization_mean:
            atom_counter = 0
            n_mean = numpy.zeros(len(chemical_properties_), dtype='float32')
        for i in range(smiles.shape[0]):
            molecule = Chem.MolFromSmiles(smiles[i].decode('utf-8'))
            AllChem.Compute2DCoords(molecule)
            for atom in molecule.GetAtoms():
                position = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
                min_x = misc.minimum(min_x, position.x)
                max_x = misc.maximum(max_x, position.x)
                min_y = misc.minimum(min_y, position.y)
                max_y = misc.maximum(max_y, position.y)
                if with_atom_symbols:
                    symbols.add(atom.GetSymbol())
                if normalization_min or normalization_max or normalization_mean:
                    chemical_property_values = chemical_properties.get_chemical_properties(atom, chemical_properties_)
                    for j in range(len(chemical_property_values)):
                        if normalization_min:
                            n_min[j] = misc.minimum(n_min[j], chemical_property_values[j])
                        if normalization_max:
                            n_max[j] = misc.maximum(n_max[j], chemical_property_values[j])
                        if normalization_mean:
                            n_mean[j] += chemical_property_values[j]
                if normalization_mean:
                    atom_counter += 1
            if with_bonds:
                for bond in molecule.GetBonds():
                    symbols.add(bond_symbols.get_bond_symbol(bond.GetBondType()))
        results['min_x'] = min_x
        results['max_x'] = max_x
        results['min_y'] = min_y
        results['max_y'] = max_y
        results['symbols'] = symbols
        if normalization_min:
            results['normalization_min'] = n_min
        if normalization_max:
            results['normalization_max'] = n_max
        if normalization_mean:
            n_mean /= atom_counter
            results['normalization_mean'] = n_mean
        return results

    @staticmethod
    def second_run(smiles, chemical_properties_, means):
        results = dict()
        n_std = numpy.zeros(len(chemical_properties_), dtype='float32')
        atom_counter = 0
        for i in range(smiles.shape[0]):
            molecule = Chem.MolFromSmiles(smiles[i].decode('utf-8'))
            for atom in molecule.GetAtoms():
                chemical_property_values = chemical_properties.get_chemical_properties(atom, chemical_properties_)
                for j in range(len(chemical_property_values)):
                    n_std[j] += (chemical_property_values[j] - means[j]) ** 2
                atom_counter += 1
        results['normalization_std'] = n_std
        results['atom_counter'] = atom_counter
        return results

    @staticmethod
    def string_list_to_numpy_array(string_list):
        max_length = 0
        for string in string_list:
            max_length = max(max_length, len(string))
        array = numpy.zeros(len(string_list), 'S' + str(max_length))
        for i in range(len(string_list)):
            array[i] = string_list[i].encode('utf-8')
        return array
