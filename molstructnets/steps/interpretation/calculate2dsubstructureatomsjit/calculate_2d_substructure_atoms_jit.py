import h5py
from rdkit import Chem

from util import data_validation, file_structure, file_util, progressbar, hdf5_util, logger, thread_pool, misc
from steps.preprocessing.shared.tensor2d import tensor_2d_jit_array
import numpy
import queue


class Calculate2DSubstructureAtomsJit:

    @staticmethod
    def get_id():
        return 'calculate_2d_substructure_atoms_jit'

    @staticmethod
    def get_name():
        return 'Calculate 2D Substructure Atoms JIT'

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
        attention_map_h5 = h5py.File(attention_map_path, 'a')
        if file_structure.Cam.substructure_atoms in attention_map_h5.keys():
            logger.log('Skipping step: ' + file_structure.Cam.substructure_atoms + ' in ' + attention_map_path
                       + ' already exists')
            attention_map_h5.close()
        else:
            attention_map_h5.close()
            temp_attention_map_path = file_util.get_temporary_file_path('attention_map')
            if file_existed:
                file_util.copy_file(attention_map_path, temp_attention_map_path)
            else:
                file_util.remove_file(attention_map_path)
            attention_map_h5 = h5py.File(temp_attention_map_path, 'a')
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles = data_h5[file_structure.DataSet.smiles][:]
            data_h5.close()
            if local_parameters['substructures'] is not None:
                substructures = local_parameters['substructures']
            else:
                substructures = hdf5_util.get_property(file_structure.get_target_file(global_parameters),
                                                       'substructures')
            substructures = substructures.split(';')
            preprocessed = tensor_2d_jit_array.load_array(global_parameters)
            substructure_atoms = hdf5_util.create_dataset(attention_map_h5,
                                                          file_structure.Cam.substructure_atoms,
                                                          (len(smiles), preprocessed.shape[1], preprocessed.shape[2]),
                                                          dtype='uint8')
            for i in range(len(substructures)):
                substructures[i] = Chem.MolFromSmiles(substructures[i], sanitize=False)
            location_queue = queue.Queue(1000)
            with progressbar.ProgressBar(len(smiles)) as progress:
                with thread_pool.ThreadPool(1) as pool:
                    pool.submit(generate_locations, preprocessed, substructures, location_queue)
                    write_substructure_atoms(location_queue, substructure_atoms, progress)
            preprocessed.close()
            attention_map_h5.close()
            file_util.move_file(temp_attention_map_path, attention_map_path)


def write_substructure_atoms(location_queue, substructure_atoms, progress):
    for i in range(substructure_atoms.shape[0]):
        atom_locations = location_queue.get()
        result = numpy.zeros((substructure_atoms.shape[1], substructure_atoms.shape[2]), dtype='uint8')
        for index in range(len(atom_locations)):
            x = atom_locations[index][0]
            y = atom_locations[index][1]
            result[x, y] = 1.0
        substructure_atoms[i, :] = result[:]
        progress.increment()


def generate_locations(preprocessed, substructures, locations_queue):
    chunks = misc.chunk_by_size(len(preprocessed), 1000)
    for chunk in chunks:
        results = preprocessed.get_substructure_locations(slice(chunk['start'], chunk['end']+1), substructures)
        for result in results:
            locations_queue.put(result)
