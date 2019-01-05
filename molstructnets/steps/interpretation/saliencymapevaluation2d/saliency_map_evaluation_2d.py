import h5py
import numpy
from rdkit import Chem

from steps.preprocessing.shared.tensor2d import tensor_2d_array
from util import data_validation, file_structure, file_util, logger, misc, progressbar, buffered_queue, hdf5_util


class SaliencyMapEvaluation2D:

    @staticmethod
    def get_id():
        return 'saliency_map_evaluation_2d'

    @staticmethod
    def get_name():
        return 'Saliency Map Evaluation'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'with_bonds', 'name': 'With bonds', 'type': bool, 'default': True,
                           'description': 'If true the activation values of bonds will also be considered. Otherwise'
                                          ' only the atoms will be used. Default: True'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_preprocessed_specs(global_parameters)
        data_validation.validate_saliency_map(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        saliency_map_h5 = h5py.File(file_structure.get_saliency_map_file(global_parameters), 'r')
        array = tensor_2d_array.load_array(global_parameters)
        if file_structure.SaliencyMap.saliency_map_active in saliency_map_h5.keys():
            saliency_map_evaluation_active_path = file_util.resolve_subpath(
                file_structure.get_interpretation_folder(global_parameters), 'saliency_map_evaluation_active.h5')
            if file_util.file_exists(saliency_map_evaluation_active_path):
                logger.log('Skipping: ' + saliency_map_evaluation_active_path + ' already exists')
            else:
                indices = None
                substructures = hdf5_util.get_property(file_structure.get_target_file(global_parameters),
                                                       'substructures')
                substructures = substructures.split(';')
                for i in range(len(substructures)):
                    substructures[i] = Chem.MolFromSmiles(substructures[i], sanitize=False)
                if file_structure.SaliencyMap.saliency_map_active_indices in saliency_map_h5.keys():
                    indices = saliency_map_h5[file_structure.SaliencyMap.saliency_map_active_indices][:]
                temp_saliency_map_evaluation_active_path = file_util.get_temporary_file_path('saliency_map_evaluation_active')
                saliency_map_active = saliency_map_h5[file_structure.SaliencyMap.saliency_map_active]
                SaliencyMapEvaluation2D.calculate_saliency_map_evaluation(saliency_map_active, array, temp_saliency_map_evaluation_active_path,
                                                                          substructures, indices, not local_parameters['with_bonds'])
                file_util.move_file(temp_saliency_map_evaluation_active_path, saliency_map_evaluation_active_path)
        if file_structure.SaliencyMap.saliency_map_inactive in saliency_map_h5.keys():
            saliency_map_evaluation_inactive_path = file_util.resolve_subpath(
                file_structure.get_interpretation_folder(global_parameters), 'saliency_map_evaluation_inactive.h5')
            if file_util.file_exists(saliency_map_evaluation_inactive_path):
                logger.log('Skipping: ' + saliency_map_evaluation_inactive_path + ' already exists')
            else:
                indices = None
                substructures = hdf5_util.get_property(file_structure.get_target_file(global_parameters),
                                                       'substructures')
                substructures = substructures.split(';')
                for i in range(len(substructures)):
                    substructures[i] = Chem.MolFromSmiles(substructures[i], sanitize=False)
                if file_structure.SaliencyMap.saliency_map_inactive_indices in saliency_map_h5.keys():
                    indices = saliency_map_h5[file_structure.SaliencyMap.saliency_map_inactive_indices][:]
                temp_saliency_map_evaluation_inactive_path = file_util.get_temporary_file_path(
                    'saliency_map_evaluation_inactive')
                saliency_map_inactive = saliency_map_h5[file_structure.SaliencyMap.saliency_map_inactive]
                SaliencyMapEvaluation2D.calculate_saliency_map_evaluation(saliency_map_inactive, array, temp_saliency_map_evaluation_inactive_path,
                                                                          substructures, indices, not local_parameters['with_bonds'])
                file_util.move_file(temp_saliency_map_evaluation_inactive_path, saliency_map_evaluation_inactive_path)
        array.close()
        saliency_map_h5.close()

    @staticmethod
    def calculate_saliency_map_evaluation(saliency_map, preprocessed, evaluation_path, substructures, indices, only_atoms):
        if indices is None:
            indices = range(len(preprocessed))
        evaluation_h5 = h5py.File(evaluation_path, 'w')
        substructure_mean = hdf5_util.create_dataset(evaluation_h5, 'substructure_mean', (len(indices),))
        substructure_std = hdf5_util.create_dataset(evaluation_h5, 'substructure_std', (len(indices),))
        not_substructure_mean = hdf5_util.create_dataset(evaluation_h5, 'not_substructure_mean', (len(indices),))
        not_substructure_std = hdf5_util.create_dataset(evaluation_h5, 'not_substructure_std', (len(indices),))
        size = misc.max_in_memory_chunk_size(saliency_map.dtype, saliency_map.shape, use_swap=False)
        location_queue = buffered_queue.BufferedQueue(1000, 10000)
        chunks = misc.chunk_by_size(len(indices), size)
        with progressbar.ProgressBar(len(indices)) as progress:
            for chunk in chunks:
                preprocessed.calc_substructure_locations(chunk['start'], chunk['end'], substructures, location_queue,
                                                         False, only_atoms)
                saliency_map_chunk = saliency_map[indices[chunk['start']:chunk['end']]]
                tmp_substructure_mean = numpy.full((len(saliency_map_chunk),), numpy.nan, dtype='float32')
                tmp_substructure_std = numpy.full((len(saliency_map_chunk),), numpy.nan, dtype='float32')
                tmp_not_substructure_mean = numpy.full((len(saliency_map_chunk),), numpy.nan, dtype='float32')
                tmp_not_substructure_std = numpy.full((len(saliency_map_chunk),), numpy.nan, dtype='float32')
                for i in range(chunk['size']):
                    index, substructure_locations, other_locations = location_queue.get()
                    if len(substructure_locations) > 0:
                        substructure_locations = list(numpy.transpose(numpy.array(substructure_locations)))
                        saliency_map_substructure = saliency_map_chunk[index][substructure_locations]
                        tmp_substructure_mean[index] = saliency_map_substructure.mean()
                        tmp_substructure_std[index] = saliency_map_substructure.std()
                    if len(other_locations) > 0:
                        other_locations = list(numpy.transpose(numpy.array(other_locations)))
                        saliency_map_other = saliency_map_chunk[index][other_locations]
                        tmp_not_substructure_mean[index] = saliency_map_other.mean()
                        tmp_not_substructure_std[index] = saliency_map_other.std()
                    progress.increment()
                substructure_mean[chunk['start']:chunk['end']] = tmp_substructure_mean[:]
                substructure_std[chunk['start']:chunk['end']] = tmp_substructure_std[:]
                not_substructure_mean[chunk['start']:chunk['end']] = tmp_not_substructure_mean[:]
                not_substructure_std[chunk['start']:chunk['end']] = tmp_not_substructure_std[:]
        evaluation_h5.close()
