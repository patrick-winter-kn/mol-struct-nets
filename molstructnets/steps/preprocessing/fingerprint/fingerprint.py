import h5py
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy

from util import data_validation, misc, file_structure, file_util, logger, progressbar, thread_pool, constants,\
    hdf5_util


number_threads = thread_pool.default_number_threads


class Fingerprint:

    @staticmethod
    def get_id():
        return 'fingerprint'

    @staticmethod
    def get_name():
        return 'Fingerprint'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'radius', 'name': 'Radius', 'type': int, 'default': 2, 'min': 1,
                           'description': 'The radius of the fingerprint. Default: 2'})
        parameters.append({'id': 'nr_bits', 'name': 'Number of bits', 'type': int, 'default': 1024, 'min': 2,
                           'description': 'The number of bits of the fingerprint. Default: 1024'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(local_parameters, ['radius', 'nr_bits'])
        file_name = 'fingerprint_' + misc.hash_parameters(hash_parameters) + '.h5'
        return file_util.resolve_subpath(file_structure.get_preprocessed_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        preprocessed_path = Fingerprint.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.preprocessed_data] = preprocessed_path
        if file_util.file_exists(preprocessed_path):
            logger.log('Skipping step: ' + preprocessed_path + ' already exists')
            preprocessed_h5 = h5py.File(preprocessed_path, 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            global_parameters[constants.GlobalParameters.input_dimensions] = (preprocessed.shape[1])
            preprocessed_h5.close()
        else:
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles_data = data_h5[file_structure.DataSet.smiles]
            temp_preprocessed_path = file_util.get_temporary_file_path('fingerprint')
            preprocessed_h5 = h5py.File(temp_preprocessed_path, 'w')
            chunks = misc.chunk(len(smiles_data), number_threads)
            global_parameters[constants.GlobalParameters.input_dimensions] = (local_parameters['nr_bits'])
            preprocessed = hdf5_util.create_dataset(preprocessed_h5, file_structure.Preprocessed.preprocessed,
                                                    (len(smiles_data), local_parameters['nr_bits']), dtype='I',
                                                    chunks=(1, local_parameters['nr_bits']))
            logger.log('Writing fingerprints')
            with progressbar.ProgressBar(len(smiles_data)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(Fingerprint.write_fingerprints, preprocessed,
                                    smiles_data[chunk['start']:chunk['end'] + 1], chunk['start'],
                                    local_parameters['radius'], local_parameters['nr_bits'], progress)
                    pool.wait()
            data_h5.close()
            preprocessed_h5.close()
            file_util.move_file(temp_preprocessed_path, preprocessed_path)

    @staticmethod
    def write_fingerprints(preprocessed, smiles_data, offset, radius, nr_bits, progress):
        for i in range(len(smiles_data)):
            smiles = smiles_data[i].decode('utf-8')
            molecule = Chem.MolFromSmiles(smiles)
            fingerprint = numpy.array(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nr_bits))
            preprocessed[i + offset, :] = fingerprint[:]
            progress.increment()
