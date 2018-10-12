import h5py
import numpy
from rdkit import Chem
from rdkit.Chem import AllChem

from steps.preprocessing.shared.chemicalproperties import chemical_properties
from steps.preprocessing.shared.tensor2d import molecule_2d_tensor, bond_symbols, rasterizer, tensor_2d_preprocessor
from util import data_validation, misc, file_structure, file_util, logger, process_pool, hdf5_util, normalization, \
    constants, multi_process_progressbar

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
        parameters.append({'id': 'bonds', 'name': 'With Bonds', 'type': bool, 'default': True,
                           'description': 'Add symbols for the bonds. Default: True'})
        parameters.append({'id': 'atom_symbols', 'name': 'With Atom Symbols', 'type': bool,
                           'default': True, 'description': 'Adds atom symbols to the data. Default: True'})
        parameters.append({'id': 'chemical_properties', 'name': 'Chemical Properties', 'type': list,
                           'default': [], 'options': chemical_properties.Properties.all,
                           'description': 'The chemical properties that will be used. Default: None'})
        parameters.append({'id': 'normalization', 'name': 'Normalization Type', 'type': str, 'default': None,
                           'options': ['None',
                                       normalization.NormalizationTypes.min_max_1,
                                       normalization.NormalizationTypes.min_max_2,
                                       normalization.NormalizationTypes.z_score],
                           'description': 'Normalization type. Default: None'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        if isinstance(global_parameters[constants.GlobalParameters.data_set], list):
            for data_set in global_parameters[constants.GlobalParameters.data_set]:
                tmp_global_parameters = global_parameters.copy()
                tmp_global_parameters[constants.GlobalParameters.data_set] = data_set
                data_validation.validate_data_set(tmp_global_parameters)
        else:
            data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(local_parameters,
                                                   ['scale', 'symbols', 'square', 'bonds', 'chemical_properties',
                                                    'normalization'])
        hash_parameters['data_set'] = global_parameters[constants.GlobalParameters.data_set]
        file_name = 'tensor_2d_jit_' + misc.hash_parameters(hash_parameters) + '.h5'
        tmp_global_parameters = global_parameters.copy()
        tmp_global_parameters[constants.GlobalParameters.data_set] = '_'.join(global_parameters[constants.GlobalParameters.data_set])
        return file_util.resolve_subpath(file_structure.get_preprocessed_folder(tmp_global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        global_parameters[constants.GlobalParameters.feature_id] = '2d_tensor'
        preprocessed_path = Tensor2D.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.preprocessed_data] = preprocessed_path
        if file_util.file_exists(preprocessed_path):
            global_parameters[constants.GlobalParameters.input_dimensions] = \
                tuple(hdf5_util.get_property(preprocessed_path, file_structure.PreprocessedTensor2D.dimensions))
            logger.log('Skipping step: ' + preprocessed_path + ' already exists')
        else:
            if isinstance(global_parameters[constants.GlobalParameters.data_set], list):
                # Concatenate all data sets in the list into one
                smiles = numpy.ndarray(shape=[0], dtype='|S0')
                for i in range(len(global_parameters[constants.GlobalParameters.data_set])):
                    data_set_h5 = h5py.File(file_structure.get_data_set_file(global_parameters, n=i), 'r')
                    smiles = numpy.concatenate((smiles, data_set_h5[file_structure.DataSet.smiles][:]))
                    data_set_h5.close()
            else:
                data_set_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
                smiles = data_set_h5[file_structure.DataSet.smiles][:]
                data_set_h5.close()
            temp_preprocessed_path = file_util.get_temporary_file_path('tensor_2d')
            preprocessed_h5 = h5py.File(temp_preprocessed_path, 'w')
            number_chemical_properties = len(local_parameters['chemical_properties'])
            chunks = misc.chunk(smiles.shape[0], process_pool.default_number_processes)
            # First run: Calculate gridsize_x, gridsize_y, symbols, normalization_min, normalization_max,
            # normalization_mean
            logger.log('Calculating stats')
            with process_pool.ProcessPool(len(chunks)) as pool:
                needs_min_max = local_parameters['normalization'] == normalization.NormalizationTypes.min_max_1 \
                                or local_parameters['normalization'] == normalization.NormalizationTypes.min_max_2
                needs_mean_std = local_parameters['normalization'] == normalization.NormalizationTypes.z_score
                with multi_process_progressbar.MultiProcessProgressbar(smiles.shape[0], value_buffer=10) as progress:
                    for chunk in chunks:
                        pool.submit(Tensor2D.first_run, smiles[chunk['start']:chunk['end']],
                                    chemical_properties_=local_parameters['chemical_properties'],
                                    with_atom_symbols=local_parameters['atom_symbols'],
                                    with_bonds=local_parameters['bonds'],
                                    normalization_min=needs_min_max, normalization_max=needs_min_max,
                                    normalization_mean=needs_mean_std, progress=progress.get_slave())
                    results = pool.get_results()
                same_values = results[0]['same_values']
                for i in range(1, len(results)):
                    for j in range(len(same_values)):
                        same_values[j].add_same_value(results[i]['same_values'][j])
                valid_property_indices = list()
                valid_properties = list()
                for i in range(len(local_parameters['chemical_properties'])):
                    if same_values[i].same():
                        logger.log('All values for ' + local_parameters['chemical_properties'][i] + ' are '
                                   + str(same_values[i].get_value()) + '. Leaving it out.', logger.LogLevel.WARNING)
                    else:
                        valid_property_indices.append(i)
                        valid_properties.append(local_parameters['chemical_properties'][i])
                min_x = None
                max_x = None
                min_y = None
                max_y = None
                symbols = set()
                if needs_min_max:
                    mins = numpy.zeros(len(valid_property_indices), dtype='float32')
                    mins[:] = numpy.nan
                    maxs = numpy.zeros(len(valid_property_indices), dtype='float32')
                    maxs[:] = numpy.nan
                if needs_mean_std:
                    means = numpy.zeros(len(valid_property_indices), dtype='float32')
                for i in range(len(results)):
                    min_x = misc.minimum(min_x, results[i]['min_x'])
                    max_x = misc.maximum(max_x, results[i]['max_x'])
                    min_y = misc.minimum(min_y, results[i]['min_y'])
                    max_y = misc.maximum(max_y, results[i]['max_y'])
                    symbols = symbols.union(results[i]['symbols'])
                    if needs_min_max or needs_mean_std:
                        index = 0
                        for j in valid_property_indices:
                            if needs_min_max:
                                mins[index] = misc.minimum(mins[index], results[i]['normalization_min'][j])
                                maxs[index] = misc.maximum(maxs[index], results[i]['normalization_max'][j])
                            if needs_mean_std:
                                means[index] += results[i]['normalization_mean'][j]
                            index += 1
                if local_parameters['symbols'] is not None:
                    symbols = symbols.union(set(local_parameters['symbols'].split(';')))
                if needs_mean_std:
                    means /= len(chunks)
                if len(symbols) > 0:
                    symbols = Tensor2D.string_list_to_numpy_array(sorted(symbols))
                    hdf5_util.create_dataset_from_data(preprocessed_h5, file_structure.PreprocessedTensor2D.symbols,
                                                       symbols)
                if needs_min_max:
                    hdf5_util.create_dataset_from_data(preprocessed_h5,
                                                       file_structure.PreprocessedTensor2D.normalization_min, mins)
                    hdf5_util.create_dataset_from_data(preprocessed_h5,
                                                       file_structure.PreprocessedTensor2D.normalization_max, maxs)
                if needs_mean_std:
                    hdf5_util.create_dataset_from_data(preprocessed_h5,
                                                       file_structure.PreprocessedTensor2D.normalization_mean, means)
                rasterizer_ = rasterizer.Rasterizer(local_parameters['scale'], tensor_2d_preprocessor.padding,
                                                    min_x, max_x,
                                                    min_y, max_y, local_parameters['square'])
                dimensions = (rasterizer_.size_x, rasterizer_.size_y, len(symbols) + len(valid_properties))
                # Second run: Calculate normalization_std
                if needs_mean_std:
                    logger.log('Calculating standard deviation')
                    with multi_process_progressbar.MultiProcessProgressbar(smiles.shape[0],
                                                                           value_buffer=10) as progress:
                        for chunk in chunks:
                            pool.submit(Tensor2D.second_run, smiles[chunk['start']:chunk['end']],
                                        valid_properties, means, progress=progress.get_slave())
                        results = pool.get_results()
                    stds = numpy.zeros(len(valid_properties), dtype='float32')
                    atom_counter = 0
                    for i in range(len(results)):
                        atom_counter += results[i]['atom_counter']
                        for j in range(len(valid_properties)):
                            stds[j] += results[i]['normalization_std'][j]
                    stds /= atom_counter
                    stds = numpy.sqrt(stds)
                    hdf5_util.create_dataset_from_data(preprocessed_h5,
                                                       file_structure.PreprocessedTensor2D.normalization_std, stds)
            # Write chemical properties
            if number_chemical_properties > 0:
                chemical_properties_array = Tensor2D.string_list_to_numpy_array(valid_properties)
                hdf5_util.create_dataset_from_data(preprocessed_h5,
                                                   file_structure.PreprocessedTensor2D.chemical_properties,
                                                   chemical_properties_array)
            hdf5_util.set_property(preprocessed_h5, file_structure.PreprocessedTensor2D.dimensions, dimensions)
            hdf5_util.set_property(preprocessed_h5, file_structure.PreprocessedTensor2D.min_x, min_x)
            hdf5_util.set_property(preprocessed_h5, file_structure.PreprocessedTensor2D.max_x, max_x)
            hdf5_util.set_property(preprocessed_h5, file_structure.PreprocessedTensor2D.min_y, min_y)
            hdf5_util.set_property(preprocessed_h5, file_structure.PreprocessedTensor2D.max_y, max_y)
            hdf5_util.set_property(preprocessed_h5, file_structure.PreprocessedTensor2D.scale,
                                   local_parameters['scale'])
            hdf5_util.set_property(preprocessed_h5, file_structure.PreprocessedTensor2D.with_bonds,
                                   local_parameters['bonds'])
            hdf5_util.set_property(preprocessed_h5, file_structure.PreprocessedTensor2D.square,
                                   local_parameters['square'])
            if local_parameters['normalization'] is not None:
                hdf5_util.set_property(preprocessed_h5, file_structure.PreprocessedTensor2D.normalization_type,
                                       local_parameters['normalization'])
            preprocessed_h5.close()
            file_util.move_file(temp_preprocessed_path, preprocessed_path)
            global_parameters[constants.GlobalParameters.input_dimensions] = dimensions

    @staticmethod
    def first_run(smiles, chemical_properties_=[], with_atom_symbols=False, with_bonds=False, normalization_min=False,
                  normalization_max=False, normalization_mean=False, progress=None):
        results = dict()
        same_values = list()
        for i in range(len(chemical_properties_)):
            same_values.append(SameValues())
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
                chemical_property_values = chemical_properties.get_chemical_properties(atom, chemical_properties_)
                for j in range(len(chemical_property_values)):
                    same_values[j].add(chemical_property_values[j])
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
            if progress is not None:
                progress.increment()
        if progress is not None:
            progress.finish()
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
        results['same_values'] = same_values
        return results

    @staticmethod
    def second_run(smiles, chemical_properties_, means, progress=None):
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
            if progress is not None:
                progress.increment()
        if progress is not None:
            progress.finish()
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


class SameValues():

    def __init__(self):
        self._value = None
        self._same = True

    def add(self, value):
        if self._same:
            if self._value is None:
                self._value = value
            elif self._value != value:
                self._same = False
                self._value = None

    def add_same_value(self, value):
        if not self._same or not value._same:
            self._same = False
            self._value = None
        elif self._value is None:
            self._value = value._value
        elif self._value != value._value:
            self._same = False
            self._value = None

    def same(self):
        return self._same

    def get_value(self):
        return self._value
