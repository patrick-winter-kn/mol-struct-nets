import h5py
import numpy
from rdkit import Chem

from steps.featuregeneration.shared import substructure_feature_generator
from steps.featuregeneration.mossfeaturegeneration import moss_integration
from util import data_validation, misc, file_structure, file_util, logger, process_pool, constants, \
    hdf5_util, multi_process_progressbar


class MossFeatureGeneration:

    @staticmethod
    def get_id():
        return 'moss_feature_generation'

    @staticmethod
    def get_name():
        return 'MoSS Feature Generation'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'min_focus', 'name': 'Minimum focus support', 'type': int, 'default': 10, 'min': 0,
                           'max': 100,
                           'description': 'The minimum (in percent) of how many active molecules need to contain a'
                                          ' substructure, for it to be valid. Default: 10'})
        parameters.append({'id': 'max_complement', 'name': 'Maximum complement support', 'type': int, 'default': 5,
                           'min': 0, 'max': 100,
                           'description': 'The maximum (in percent) of how many inactive molecules can contain a'
                                          ' substructure, for it to be valid. Default: 5'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        file_name = 'moss_features.h5'
        return file_util.resolve_subpath(file_structure.get_result_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        global_parameters[constants.GlobalParameters.feature_id] = 'moss_substructures'
        features_path = MossFeatureGeneration.get_result_file(global_parameters, local_parameters)
        if file_util.file_exists(features_path):
            logger.log('Skipping step: ' + features_path + ' already exists')
            features_h5 = h5py.File(features_path, 'r')
            feature_dimensions = features_h5[file_structure.Preprocessed.preprocessed].shape[1]
            features_h5.close()
        else:
            partition_h5 = h5py.File(file_structure.get_partition_file(global_parameters), 'r')
            train_indices = numpy.unique(partition_h5[file_structure.Partitions.train][:])
            partition_h5.close()
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles_train_data = numpy.take(data_h5[file_structure.DataSet.smiles][:], train_indices, axis=0)
            smiles_data = data_h5[file_structure.DataSet.smiles][:]
            data_h5.close()
            target_h5 = h5py.File(file_structure.get_target_file(global_parameters), 'r')
            train_classes = numpy.take(target_h5[file_structure.Target.classes][:], train_indices, axis=0)
            target_h5.close()
            substructures = moss_integration.calculate_substructures(smiles_train_data, train_classes,
                                                                     local_parameters['min_focus']/100,
                                                                     local_parameters['max_complement']/100)
            feature_dimensions = len(substructures)
            temp_features_path = file_util.get_temporary_file_path('moss_features')
            chunks = misc.chunk(len(smiles_data), process_pool.default_number_processes)
            global_parameters[constants.GlobalParameters.input_dimensions] = (len(substructures),)
            logger.log('Calculating MoSS features')
            with process_pool.ProcessPool(len(chunks)) as pool:
                with multi_process_progressbar.MultiProcessProgressbar(len(smiles_data), value_buffer=100) as progress:
                    for chunk in chunks:
                        pool.submit(substructure_feature_generator.generate_substructure_features,
                                    smiles_data[chunk['start']:chunk['end']], substructures,
                                    progress=progress.get_slave())
                    results = pool.get_results()
            features_h5 = h5py.File(temp_features_path, 'w')
            features = hdf5_util.create_dataset(features_h5, file_structure.Preprocessed.preprocessed,
                                                (len(smiles_data), len(substructures)), dtype='uint16',
                                                chunks=(1, len(substructures)))
            offset = 0
            for result in results:
                features[offset:offset + len(result)] = result[:]
                offset += len(result)
            features_h5.close()
            file_util.move_file(temp_features_path, features_path)
        global_parameters[constants.GlobalParameters.input_dimensions] = feature_dimensions
        global_parameters[constants.GlobalParameters.preprocessed_data] = features_path
