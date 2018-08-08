import h5py
import numpy
from rdkit import Chem

from steps.featuregeneration.shared import substructure_feature_generator
from util import data_validation, misc, file_structure, file_util, logger, process_pool, constants, \
    hdf5_util, multi_process_progressbar


class CamSubstructureFeatureGeneration:

    @staticmethod
    def get_id():
        return 'cam_substructure_feature_generation'

    @staticmethod
    def get_name():
        return 'CAM Substructure Feature Generation'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'top_n', 'name': 'Top n', 'type': int, 'default': None, 'min': 1,
                           'description': 'The number of substructures that will be considered. Default: All'})
        parameters.append({'id': 'min_score', 'name': 'Minimum score', 'type': float, 'default': None, 'min': 0,
                           'max': 1,
                           'description': 'The minimum score of substructures that will be considered. Default: All'})
        parameters.append({'id': 'active', 'name': 'Active class', 'type': bool, 'default': True,
                           'description': 'Use substructures for the active class (True) or inactive class (False).'
                                          ' Default: Active'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        file_name = 'cam_features.h5'
        return file_util.resolve_subpath(file_structure.get_result_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        features_path = CamSubstructureFeatureGeneration.get_result_file(global_parameters, local_parameters)
        if file_util.file_exists(features_path):
            logger.log('Skipping step: ' + features_path + ' already exists')
            features_h5 = h5py.File(features_path, 'r')
            feature_dimensions = features_h5[file_structure.Preprocessed.preprocessed].shape[1]
            features_h5.close()
        else:
            cam_substructures_path = global_parameters[constants.GlobalParameters.cam_substructures_data]
            substructures = load_substructures(cam_substructures_path, local_parameters['top_n'],
                                               local_parameters['min_score'], local_parameters['active'])
            feature_dimensions = len(substructures)
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles_data = data_h5[file_structure.DataSet.smiles][:]
            data_h5.close()
            temp_features_path = file_util.get_temporary_file_path('cam_features')
            chunks = misc.chunk(len(smiles_data), process_pool.default_number_processes)
            global_parameters[constants.GlobalParameters.input_dimensions] = (len(substructures),)
            logger.log('Calculating cam features')
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


def load_substructures(cam_substructures_path, top_n, min_score, active):
    if active:
        score_data_set = file_structure.CamSubstructures.active_substructures_score
        smiles_data_set = file_structure.CamSubstructures.active_substructures
    else:
        score_data_set = file_structure.CamSubstructures.inactive_substructures_score
        smiles_data_set = file_structure.CamSubstructures.inactive_substructures
    cam_substructures_h5 = h5py.File(cam_substructures_path, 'r')
    number_substructures = len(cam_substructures_h5[smiles_data_set])
    if top_n is not None:
        number_substructures = min(number_substructures, top_n)
    if min_score is not None:
        score = cam_substructures_h5[score_data_set][:]
        number_substructures = min(number_substructures, numpy.sum(score >= min_score))
    smiles = cam_substructures_h5[smiles_data_set][:number_substructures]
    cam_substructures_h5.close()
    substructures = list()
    for i in range(len(smiles)):
        substructures.append(Chem.MolFromSmiles(smiles[i].decode('UTF-8'), sanitize=False))
    return substructures
