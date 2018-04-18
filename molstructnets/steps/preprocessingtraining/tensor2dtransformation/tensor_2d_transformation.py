import random

import h5py
from rdkit import Chem
from steps.preprocessing.shared.tensor2d import rasterizer
from steps.preprocessingtraining.tensor2dtransformation import transformer
from util import data_validation, misc, file_structure, file_util, logger, progressbar, concurrent_set, constants,\
    hdf5_util, reference_data_set, normalization
from steps.preprocessing.shared.tensor2d import molecule_2d_tensor
import math


output_rows_per_chunk = 10000


class Tensor2DTransformed:

    @staticmethod
    def get_id():
        return 'tensor_2d_transformation'

    @staticmethod
    def get_name():
        return 'Transformed 2D Tensor'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'transformations', 'name': 'Number of Transformations', 'type': int, 'default': 1,
                           'min': 0,
                           'description': 'The number of transformations done for each tensor in the training data'
                                          ' set. Default: 1'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(global_parameters,
                                                   [constants.GlobalParameters.seed])
        scale = hdf5_util.get_property(global_parameters[constants.GlobalParameters.preprocessed_data], 'scale')
        hash_parameters['scale'] = scale
        chemical_properties = hdf5_util.get_property(global_parameters[constants.GlobalParameters.preprocessed_data],
                                                     'chemical_properties')
        hash_parameters['chemical_properties'] = chemical_properties
        preprocessed_path = global_parameters[constants.GlobalParameters.preprocessed_data]
        hash_parameters['normalize'] =\
            hdf5_util.has_data_set(preprocessed_path, file_structure.Preprocessed.preprocessed_normalization_stats)
        file_name = 'tensor_2d_transformation_' + str(local_parameters['transformations']) + '_'\
                    + misc.hash_parameters(hash_parameters) + '.h5'
        return file_util.resolve_subpath(file_structure.get_preprocessed_training_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        preprocessed_training_path = Tensor2DTransformed.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.preprocessed_training_data] = preprocessed_training_path
        rows_done = hdf5_util.get_property(preprocessed_training_path, 'rows_done')
        file_exists = file_util.file_exists(preprocessed_training_path)
        preprocessed_path = global_parameters[constants.GlobalParameters.preprocessed_data]
        needs_normalization = hdf5_util.has_data_set(preprocessed_path,
                                                     file_structure.Preprocessed.preprocessed_normalization_stats)
        if file_exists and needs_normalization:
            if hdf5_util.get_property(preprocessed_training_path, 'needs_normalization') is None:
                needs_normalization = False
        if file_exists and rows_done is None and not needs_normalization:
            logger.log('Skipping step: ' + preprocessed_training_path + ' already exists')
        else:
            if not file_exists or rows_done is not None:
                transformations = local_parameters['transformations']
                chemical_properties =\
                    eval(hdf5_util.get_property(global_parameters[constants.GlobalParameters.preprocessed_data],
                                           'chemical_properties'))
                start = 0
                if rows_done is not None:
                    start = int(rows_done / transformations)
                input_rows_per_chunk = math.ceil(output_rows_per_chunk / transformations)
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
                index_lookup = {}
                for i in range(len(index)):
                    index_lookup[index[i].decode('utf-8')] = i
                rasterizer_ = rasterizer.Rasterizer(scale, molecule_2d_tensor.padding, min_x, max_x, min_y, max_y,
                                                    preprocessed.shape[1] == preprocessed.shape[2])
                transformer_ = transformer.Transformer(min_x, max_x, min_y, max_y)
                partition_h5 = h5py.File(file_structure.get_partition_file(global_parameters), 'r')
                train = partition_h5[file_structure.Partitions.train]
                train_smiles_data = reference_data_set.ReferenceDataSet(train, smiles_data)
                temp_preprocessed_training_path = file_util.get_temporary_file_path('tensor_2d_transformation')
                if file_util.file_exists(preprocessed_training_path):
                    file_util.copy_file(preprocessed_training_path, temp_preprocessed_training_path)
                else:
                    preprocessed_training_h5 = h5py.File(temp_preprocessed_training_path, 'w')
                    number_rows = train.shape[0] * transformations
                    hdf5_util.create_dataset(preprocessed_training_h5,
                                             file_structure.PreprocessedTraining.preprocessed_training,
                                             (number_rows, preprocessed.shape[1], preprocessed.shape[2],
                                              preprocessed.shape[3]),
                                             dtype=preprocessed.dtype,
                                             chunks=(1, preprocessed.shape[1], preprocessed.shape[2],
                                                     preprocessed.shape[3]))
                    hdf5_util.create_dataset(preprocessed_training_h5,
                                             file_structure.PreprocessedTraining.preprocessed_training_references,
                                             (number_rows,), dtype='I')
                    preprocessed_training_h5.close()
                    if needs_normalization:
                        hdf5_util.set_property(temp_preprocessed_training_path, 'needs_normalization', True)
                originals_set = concurrent_set.ConcurrentSet()
                for i in range(start):
                    originals_set.add(train_smiles_data[i].decode('utf-8'))
                logger.log('Writing transformed training data')
                with progressbar.ProgressBar((len(train_smiles_data) - start) * transformations) \
                        as progress:
                    while start < len(train_smiles_data):
                        preprocessed_training_h5 = h5py.File(temp_preprocessed_training_path, 'r+')
                        preprocessed_training =\
                            preprocessed_training_h5[file_structure.PreprocessedTraining.preprocessed_training]
                        preprocessed_training_ref =\
                            preprocessed_training_h5[file_structure.PreprocessedTraining.preprocessed_training_references]
                        end = min(start + input_rows_per_chunk, len(train_smiles_data))
                        Tensor2DTransformed.\
                            write_transformed_2d_tensors(preprocessed_training, preprocessed_training_ref,
                                                         train_smiles_data, index_lookup, rasterizer_, transformer_, start,
                                                         end, originals_set, transformations,
                                                         len(train_smiles_data),
                                                         global_parameters[constants.GlobalParameters.seed], train,
                                                         chemical_properties, progress)
                        start = end
                        hdf5_util.set_property(temp_preprocessed_training_path, 'rows_done', end * transformations)
                        preprocessed_training_h5.close()
                        file_util.copy_file(temp_preprocessed_training_path, preprocessed_training_path)
                file_util.remove_file(temp_preprocessed_training_path)
                hdf5_util.delete_property(preprocessed_training_path, 'rows_done')
                data_h5.close()
                preprocessed_h5.close()
            if needs_normalization:
                preprocessed_h5 = h5py.File(preprocessed_path, 'r')
                normalization_stats = preprocessed_h5[file_structure.Preprocessed.preprocessed_normalization_stats]
                normalization.normalize_data_set(preprocessed_training_path,
                                                 file_structure.PreprocessedTraining.preprocessed_training,
                                                 stats=normalization_stats)
                preprocessed_h5.close()
                hdf5_util.delete_property(preprocessed_training_path, 'needs_normalization')

    @staticmethod
    def write_transformed_2d_tensors(preprocessed_training, preprocessed_training_ref, smiles_data, index_lookup,
                                     rasterizer_, transformer_, start, end, originals_set, number_transformations,
                                     offset_per_transformation, seed, train_ref, chemical_properties, progress):
        random_ = random.Random()
        for i in range(start, end):
            random_.seed(seed + i)
            original_index = train_ref[i]
            smiles = smiles_data[i].decode('utf-8')
            molecule = Chem.MolFromSmiles(smiles)
            for j in range(number_transformations):
                index = i + offset_per_transformation * j
                if originals_set.add(smiles):
                    preprocessed_row =\
                        molecule_2d_tensor.molecule_to_2d_tensor(molecule, index_lookup, rasterizer_,
                                                                 preprocessed_training.shape,
                                                                 chemical_properties_=chemical_properties)[0]
                else:
                    preprocessed_row =\
                        molecule_2d_tensor.molecule_to_2d_tensor(molecule, index_lookup, rasterizer_,
                                                                 preprocessed_training.shape, transformer_=transformer_,
                                                                 random_=random_,
                                                                 chemical_properties_=chemical_properties)[0]
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
