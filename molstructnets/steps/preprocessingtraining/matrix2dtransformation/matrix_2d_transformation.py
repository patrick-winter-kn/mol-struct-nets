import random

import h5py
from rdkit import Chem
from steps.preprocessing.shared.matrix2d import rasterizer
from steps.preprocessingtraining.matrix2dtransformation import transformer
from util import data_validation, misc, file_structure, file_util, logger, progressbar, concurrent_set, thread_pool,\
    constants, hdf5_util, reference_data_set
from steps.preprocessing.shared.matrix2d import molecule_2d_matrix


class Matrix2DTransformed:

    @staticmethod
    def get_id():
        return 'matrix_2d_transformation'

    @staticmethod
    def get_name():
        return 'Transformed 2D Matrix'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'transformations', 'name': 'Number of transformations for each matrix (default: 1)',
                           'type': int, 'default': 1,
                           'description': 'The number of transformations done for each matrix in the training data'
                                          ' set.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(global_parameters, [constants.GlobalParameters.seed])
        file_name = 'matrix_2d_transformation_' + str(local_parameters['transformations']) + '_'\
                    + misc.hash_parameters(hash_parameters) + '.h5'
        return file_util.resolve_subpath(file_structure.get_preprocessed_training_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        preprocessed_training_path = Matrix2DTransformed.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.preprocessed_training_data] = preprocessed_training_path
        if file_util.file_exists(preprocessed_training_path):
            logger.log('Skipping step: ' + preprocessed_training_path + ' already exists')
        else:
            preprocessed_path = global_parameters[constants.GlobalParameters.preprocessed_data]
            scale = hdf5_util.get_property(preprocessed_path, 'scale')
            min_x = hdf5_util.get_property(preprocessed_path, 'min_x')
            max_x = hdf5_util.get_property(preprocessed_path, 'max_x')
            min_y = hdf5_util.get_property(preprocessed_path, 'min_y')
            max_y = hdf5_util.get_property(preprocessed_path, 'max_y')
            preprocessed_h5 = h5py.File(preprocessed_path, 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            index = preprocessed_h5[file_structure.Preprocessed.index]
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles_data = data_h5[file_structure.DataSet.smiles]
            temp_preprocessed_training_path = file_util.get_temporary_file_path('matrix_2d_transformation')
            preprocessed_training_h5 = h5py.File(temp_preprocessed_training_path, 'w')
            index_lookup = {}
            for i in range(len(index)):
                index_lookup[index[i].decode('utf-8')] = i
            rasterizer_ = rasterizer.Rasterizer(scale, molecule_2d_matrix.padding, min_x, max_x, min_y, max_y,
                                                preprocessed.shape[1] == preprocessed.shape[2])
            transformer_ = transformer.Transformer(min_x, max_x, min_y, max_y)
            logger.log('Writing transformed training data')
            partition_h5 = h5py.File(file_structure.get_partition_file(global_parameters), 'r')
            train = partition_h5[file_structure.Partitions.train]
            preprocessed_training = \
                hdf5_util.create_dataset(preprocessed_training_h5,
                                         file_structure.PreprocessedTraining.preprocessed_training,
                                         (train.shape[0] * local_parameters['transformations'],
                                          preprocessed.shape[1], preprocessed.shape[2], preprocessed.shape[3]),
                                         dtype='I', chunks=(1, preprocessed.shape[1], preprocessed.shape[2],
                                                            preprocessed.shape[3]))
            preprocessed_training_ref = \
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
                        pool.submit(Matrix2DTransformed.write_transformed_2d_matrices, preprocessed_training,
                                    preprocessed_training_ref, train_smiles_data[chunk['start']:chunk['end'] + 1],
                                    index_lookup, rasterizer_, transformer_, chunk['start'],
                                    originals_set, local_parameters['transformations'], len(train_smiles_data),
                                    global_parameters[constants.GlobalParameters.seed] + chunk['start'], train,
                                    progress)
                    pool.wait()
            data_h5.close()
            preprocessed_h5.close()
            preprocessed_training_h5.close()
            file_util.move_file(temp_preprocessed_training_path, preprocessed_training_path)

    @staticmethod
    def write_transformed_2d_matrices(preprocessed_training, preprocessed_training_ref, smiles_data, index_lookup,
                                      rasterizer_, transformer_, offset, originals_set, number_transformations,
                                      offset_per_transformation, seed, train_ref, progress):
        random_ = random.Random(seed)
        for i in range(len(smiles_data)):
            original_index = train_ref[i + offset]
            smiles = smiles_data[i].decode('utf-8')
            molecule = Chem.MolFromSmiles(smiles)
            start = 0
            if originals_set.add(smiles):
                preprocessed_row = molecule_2d_matrix.molecule_to_2d_matrix(molecule, index_lookup, rasterizer_,
                                                                            preprocessed_training.shape)[0]
                preprocessed_training[i + offset, :] = preprocessed_row[:]
                preprocessed_training_ref[i + offset] = original_index
                start += 1
                progress.increment()
            for j in range(start, number_transformations):
                index = i + offset + offset_per_transformation * j
                preprocessed_row =\
                    molecule_2d_matrix.molecule_to_2d_matrix(molecule, index_lookup, rasterizer_,
                                                             preprocessed_training.shape, transformer_=transformer_,
                                                             random_=random_)[0]
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
