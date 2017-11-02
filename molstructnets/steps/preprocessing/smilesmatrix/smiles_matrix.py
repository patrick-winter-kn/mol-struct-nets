from util import data_validation, misc, file_structure, file_util, logger, progressbar, concurrent_max, concurrent_set,\
    thread_pool, constants, hdf5_util, reference_data_set
import numpy
import h5py
from rdkit import Chem
import random


number_threads = thread_pool.default_number_threads


class SmilesMatrix:

    @staticmethod
    def get_id():
        return 'smiles_matrix'

    @staticmethod
    def get_name():
        return 'SMILES Matrix'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'max_length', 'name': 'Maximum length (default: automatic)', 'type': int,
                           'default': None,
                           'description': 'Maximum number of characters of input SMILES string. If the limit is'
                                          ' exceeded the string will be shortened to fit into the matrix.'})
        parameters.append({'id': 'characters', 'name': 'Force characters (default: none)', 'type': str,
                           'default': None, 'description': 'Characters in the given string will be added to the index'
                                                           ' in addition to characters found in the data set.'})
        parameters.append({'id': 'transformations', 'name': 'Number of transformations for each SMILES (default: none)',
                           'type': int, 'default': 0,
                           'description': 'The number of transformations done for each SMILES string in the training'
                                          ' data set.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        if local_parameters['transformations'] > 0:
            data_validation.validate_target(global_parameters)
            data_validation.validate_partition(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(global_parameters, [constants.GlobalParameters.seed])
        hash_parameters.update(misc.copy_dict_from_keys(local_parameters, ['max_length', 'characters',
                                                                           'transformations']))
        file_name = 'smiles_matrix_' + misc.hash_parameters(hash_parameters) + '.h5'
        return file_util.resolve_subpath(file_structure.get_preprocessed_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        preprocessed_path = SmilesMatrix.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.preprocessed_data] = preprocessed_path
        if file_util.file_exists(preprocessed_path):
            logger.log('Skipping step: ' + preprocessed_path + ' already exists')
            preprocessed_h5 = h5py.File(preprocessed_path, 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            global_parameters[constants.GlobalParameters.input_dimensions] = (preprocessed.shape[1],
                                                                              preprocessed.shape[2])
            preprocessed_h5.close()
        else:
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles_data = data_h5[file_structure.DataSet.smiles]
            temp_preprocessed_path = file_util.get_temporary_file_path('smiles_matrix')
            preprocessed_h5 = h5py.File(temp_preprocessed_path, 'w')
            chunks = misc.chunk(len(smiles_data), number_threads)
            characters = concurrent_set.ConcurrentSet()
            if local_parameters['characters'] is not None:
                for character in local_parameters['characters']:
                    characters.add(character)
            max_length = concurrent_max.ConcurrentMax()
            if local_parameters['max_length'] is not None:
                max_length.add_value(local_parameters['max_length'])
            logger.log('Analyzing SMILES')
            with progressbar.ProgressBar(len(smiles_data)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(SmilesMatrix.analyze_smiles, smiles_data[chunk['start']:chunk['end'] + 1],
                                    characters, max_length, progress)
                    pool.wait()
            characters = sorted(characters.get_set_copy())
            index = hdf5_util.create_dataset(preprocessed_h5, 'index', (len(characters),), dtype='S1')
            index_lookup = {}
            for i in range(len(characters)):
                index_lookup[characters[i]] = i
                index[i] = characters[i].encode('utf-8')
            global_parameters[constants.GlobalParameters.input_dimensions] = (max_length.get_max(), len(index))
            preprocessed = hdf5_util.create_dataset(preprocessed_h5, file_structure.Preprocessed.preprocessed,
                                                          (len(smiles_data), max_length.get_max(), len(index)),
                                                          dtype='I', chunks=(1, max_length.get_max(), len(index)))
            logger.log('Writing matrices')
            with progressbar.ProgressBar(len(smiles_data)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(SmilesMatrix.write_smiles_matrices, preprocessed,
                                    smiles_data[chunk['start']:chunk['end'] + 1], index_lookup, max_length.get_max(),
                                    chunk['start'], progress)
                    pool.wait()
            if local_parameters['transformations'] > 0:
                logger.log('Writing transformed training data')
                partition_h5 = h5py.File(global_parameters[constants.GlobalParameters.partition_data], 'r')
                train = partition_h5[file_structure.Partitions.train]
                preprocessed_training =\
                    hdf5_util.create_dataset(preprocessed_h5, file_structure.Preprocessed.preprocessed_training,
                                             (train.shape[0] * local_parameters['transformations'],
                                              max_length.get_max(), len(index)),
                                             dtype='I', chunks=(1, max_length.get_max(), len(index)))
                preprocessed_training_ref =\
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
                            pool.submit(SmilesMatrix.write_transformed_smiles_matrices, preprocessed_training,
                                        preprocessed_training_ref, train_smiles_data[chunk['start']:chunk['end'] + 1],
                                        index_lookup, max_length.get_max(), chunk['start'], progress,
                                        local_parameters['transformations'], len(train_smiles_data),
                                        random.Random(global_parameters[constants.GlobalParameters.seed]
                                                      + chunk['start']), train, originals_set)
                        pool.wait()
            data_h5.close()
            preprocessed_h5.close()
            file_util.move_file(temp_preprocessed_path, preprocessed_path)

    @staticmethod
    def analyze_smiles(smiles_data, characters, max_length, progress):
        characters.add(' ')
        for string in smiles_data:
            string = string.decode('utf-8')
            max_length.add_value(len(string))
            for character in string:
                characters.add(character)
            progress.increment()

    @staticmethod
    def pad_string(string, length):
        return string + (' ' * (length - len(string)))

    @staticmethod
    def string_to_matrix(string, index_lookup):
        matrix = numpy.zeros((len(string), len(index_lookup)))
        for i in range(len(string)):
            matrix[i][index_lookup[string[i]]] = 1
        return matrix

    @staticmethod
    def write_smiles_matrices(preprocessed, smiles_data, index_lookup, max_length, offset, progress):
        for i in range(len(smiles_data)):
            string = SmilesMatrix.pad_string(smiles_data[i].decode('utf-8'), max_length)
            preprocessed[i + offset] = SmilesMatrix.string_to_matrix(string, index_lookup)
            progress.increment()

    @staticmethod
    def write_transformed_smiles_matrices(preprocessed_training, preprocessed_training_ref, smiles_data, index_lookup,
                                          max_length, offset, progress, number_transformations,
                                          offset_per_transformation, random_, train_ref, originals_set):
        for i in range(len(smiles_data)):
            original_index = train_ref[i + offset]
            original_smiles = smiles_data[i].decode('utf-8')
            molecule = Chem.MolFromSmiles(original_smiles)
            atom_indices = list(range(molecule.GetNumAtoms()))
            start = 0
            if originals_set.add(original_smiles):
                # Add original smiles first
                string = SmilesMatrix.pad_string(original_smiles, max_length)
                preprocessed_training[i + offset] = SmilesMatrix.string_to_matrix(string, index_lookup)
                preprocessed_training_ref[i + offset] = original_index
                start += 1
                progress.increment()
            for j in range(start, number_transformations):
                invalid = True
                while invalid:
                    random_.shuffle(atom_indices)
                    smiles = Chem.MolToSmiles(Chem.RenumberAtoms(molecule, atom_indices), canonical=False)
                    invalid = len(smiles) > max_length
                index = i + offset + offset_per_transformation * j
                string = SmilesMatrix.pad_string(smiles, max_length)
                preprocessed_training[index] = SmilesMatrix.string_to_matrix(string, index_lookup)
                preprocessed_training_ref[index] = original_index
                progress.increment()
