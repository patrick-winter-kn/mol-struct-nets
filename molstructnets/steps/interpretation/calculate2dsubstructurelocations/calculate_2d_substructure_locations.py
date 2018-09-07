import h5py
import numpy
from rdkit import Chem

from steps.preprocessing.shared.tensor2d import tensor_2d_array
from util import data_validation, file_structure, file_util, progressbar, hdf5_util, logger, buffered_queue, misc


class Calculate2DSubstructureLocations:

    @staticmethod
    def get_id():
        return 'calculate_2d_substructure_locations'

    @staticmethod
    def get_name():
        return 'Calculate 2D Substructure Locations'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'substructures', 'name': 'Substructures', 'type': str, 'default': None,
                           'description': 'Semicolon separated list of substructure to search for. If None then the'
                                          ' substructures of the target generation step are used. Default: Use'
                                          ' generated target'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_target(global_parameters)
        data_validation.validate_preprocessed_jit(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        attention_map_path = file_structure.get_cam_file(global_parameters)
        file_existed = file_util.file_exists(attention_map_path)
        file_util.make_folders(attention_map_path)
        cam_h5 = h5py.File(attention_map_path, 'a')
        if file_structure.Cam.substructure_atoms in cam_h5.keys():
            logger.log('Skipping step: ' + file_structure.Cam.substructure_atoms + ' in ' + attention_map_path
                       + ' already exists')
            cam_h5.close()
        else:
            cam_h5.close()
            temp_cam_path = file_util.get_temporary_file_path('cam')
            if file_existed:
                file_util.copy_file(attention_map_path, temp_cam_path)
            else:
                file_util.remove_file(attention_map_path)
            cam_h5 = h5py.File(temp_cam_path, 'a')
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles = data_h5[file_structure.DataSet.smiles][:]
            data_h5.close()
            if local_parameters['substructures'] is not None:
                substructures = local_parameters['substructures']
            else:
                substructures = hdf5_util.get_property(file_structure.get_target_file(global_parameters),
                                                       'substructures')
            substructures = substructures.split(';')
            preprocessed = tensor_2d_array.load_array(global_parameters)
            substructure_locations = hdf5_util.create_dataset(cam_h5,
                                                              file_structure.Cam.substructure_atoms,
                                                              (len(smiles), preprocessed.shape[1],
                                                               preprocessed.shape[2]),
                                                              dtype='uint8',
                                                              chunks=(1, preprocessed.shape[1], preprocessed.shape[2]))
            for i in range(len(substructures)):
                substructures[i] = Chem.MolFromSmiles(substructures[i], sanitize=False)
            with progressbar.ProgressBar(len(smiles)) as progress:
                write_substructure_locations(preprocessed, substructures, substructure_locations, progress)
            preprocessed.close()
            cam_h5.close()
            file_util.move_file(temp_cam_path, attention_map_path)


def write_substructure_locations(preprocessed, substructures, substructure_atoms, progress):
    size = misc.max_in_memory_chunk_size('uint8', (1, substructure_atoms.shape[1], substructure_atoms.shape[2]),
                                         use_swap=False)
    location_queue = buffered_queue.BufferedQueue(1000, 10000)
    chunks = misc.chunk_by_size(substructure_atoms.shape[0], size)
    for chunk in chunks:
        preprocessed.calc_substructure_locations(chunk['start'], chunk['end'], substructures, location_queue, True)
        result = numpy.zeros((chunk['size'], substructure_atoms.shape[1], substructure_atoms.shape[2]), dtype='uint8')
        for i in range(result.shape[0]):
            idx, atom_locations = location_queue.get()
            for index in range(len(atom_locations)):
                x = atom_locations[index][0]
                y = atom_locations[index][1]
                result[idx, x, y] = 1.0
            progress.increment()
        substructure_atoms[chunk['start']:chunk['end']] = result[:]
