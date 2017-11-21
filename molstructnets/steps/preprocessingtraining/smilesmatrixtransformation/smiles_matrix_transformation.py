from util import data_validation, misc, file_structure, file_util, logger, progressbar, concurrent_set, thread_pool,\
    constants, hdf5_util, reference_data_set
import numpy
import h5py
from rdkit import Chem
import random


class SmilesMatrixTransformation:

    @staticmethod
    def get_id():
        return 'smiles_matrix_transformation'

    @staticmethod
    def get_name():
        return 'Transformed SMILES Matrix'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'transformations', 'name': 'Number of transformations for each SMILES (default: none)',
                           'type': int, 'default': 1,
                           'description': 'The number of transformations done for each SMILES string in the training'
                                          ' data set.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(global_parameters, [constants.GlobalParameters.seed])
        file_name = 'smiles_matrix_transformation_' + str(local_parameters['transformations']) + '_'\
                    + misc.hash_parameters(hash_parameters) + '.h5'
        return file_util.resolve_subpath(file_structure.get_preprocessed_training_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        preprocessed_training_path = SmilesMatrixTransformation.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.preprocessed_training_data] = preprocessed_training_path
        if file_util.file_exists(preprocessed_training_path):
            logger.log('Skipping step: ' + preprocessed_training_path + ' already exists')
        else:
            preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data], 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            index = preprocessed_h5[file_structure.Preprocessed.index]
            index_lookup = {}
            for i in range(len(index)):
                index_lookup[index[i].decode('utf-8')] = i
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles_data = data_h5[file_structure.DataSet.smiles]
            temp_preprocessed_training_path = file_util.get_temporary_file_path('smiles_matrix_transformation')
            preprocessed_training_h5 = h5py.File(temp_preprocessed_training_path, 'w')
            logger.log('Writing transformed training data')
            partition_h5 = h5py.File(file_structure.get_partition_file(global_parameters), 'r')
            train = partition_h5[file_structure.Partitions.train]
            preprocessed_training =\
                hdf5_util.create_dataset(preprocessed_training_h5, file_structure.PreprocessedTraining.preprocessed_training,
                                         (train.shape[0] * local_parameters['transformations'],
                                          preprocessed.shape[1], len(index)),
                                         dtype='I', chunks=(1, preprocessed.shape[1], len(index)))
            preprocessed_training_ref =\
                hdf5_util.create_dataset(preprocessed_training_h5,
                                         file_structure.PreprocessedTraining.preprocessed_training_references,
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
                        pool.submit(SmilesMatrixTransformation.write_transformed_smiles_matrices, preprocessed_training,
                                    preprocessed_training_ref, train_smiles_data[chunk['start']:chunk['end'] + 1],
                                    index_lookup, preprocessed.shape[1], chunk['start'], progress,
                                    local_parameters['transformations'], len(train_smiles_data),
                                    random.Random(global_parameters[constants.GlobalParameters.seed]
                                                  + chunk['start']), train, originals_set)
                    pool.wait()
            data_h5.close()
            preprocessed_h5.close()
            preprocessed_training_h5.close()
            file_util.move_file(temp_preprocessed_training_path, preprocessed_training_path)

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
                string = SmilesMatrixTransformation.pad_string(original_smiles, max_length)
                preprocessed_training[i + offset] = SmilesMatrixTransformation.string_to_matrix(string, index_lookup)
                preprocessed_training_ref[i + offset] = original_index
                start += 1
                progress.increment()
            for j in range(start, number_transformations):
                invalid = True
                while invalid:
                    random_.shuffle(atom_indices)
                    smiles = Chem.MolToSmiles(Chem.RenumberAtoms(molecule, atom_indices), canonical=False)
                    invalid = len(smiles) > max_length or SmilesMatrixTransformation.invalid_characters(smiles,
                                                                                                        index_lookup)
                index = i + offset + offset_per_transformation * j
                string = SmilesMatrixTransformation.pad_string(smiles, max_length)
                preprocessed_training[index] = SmilesMatrixTransformation.string_to_matrix(string, index_lookup)
                preprocessed_training_ref[index] = original_index
                progress.increment()

    @staticmethod
    def invalid_characters(smiles, index_lookup):
        for character in smiles:
            if character not in index_lookup:
                return True
        return False
