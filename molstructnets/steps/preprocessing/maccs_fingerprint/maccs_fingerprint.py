import h5py
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
import numpy

from util import data_validation, misc, file_structure, file_util, logger, progressbar, thread_pool, constants,\
    hdf5_util


number_threads = thread_pool.default_number_threads


class MaccsFingerprint:

    @staticmethod
    def get_id():
        return 'maccs_fingerprint'

    @staticmethod
    def get_name():
        return 'MACCS Fingerprint'

    @staticmethod
    def get_parameters():
        return list()

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        file_name = 'maccs_fingerprint.h5'
        return file_util.resolve_subpath(file_structure.get_preprocessed_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        preprocessed_path = MaccsFingerprint.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.preprocessed_data] = preprocessed_path
        if file_util.file_exists(preprocessed_path):
            logger.log('Skipping step: ' + preprocessed_path + ' already exists')
            global_parameters[constants.GlobalParameters.input_dimensions] = (166)
        else:
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles_data = data_h5[file_structure.DataSet.smiles]
            temp_preprocessed_path = file_util.get_temporary_file_path('maccs_fingerprint')
            preprocessed_h5 = h5py.File(temp_preprocessed_path, 'w')
            chunks = misc.chunk(len(smiles_data), number_threads)
            global_parameters[constants.GlobalParameters.input_dimensions] = (166)
            preprocessed = hdf5_util.create_dataset(preprocessed_h5, file_structure.Preprocessed.preprocessed,
                                                    (len(smiles_data), 166), dtype='I', chunks=(1, 166))
            logger.log('Writing fingerprints')
            with progressbar.ProgressBar(len(smiles_data)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(MaccsFingerprint.write_fingerprints, preprocessed,
                                    smiles_data[chunk['start']:chunk['end'] + 1], chunk['start'], progress)
                    pool.wait()
            data_h5.close()
            preprocessed_h5.close()
            file_util.move_file(temp_preprocessed_path, preprocessed_path)

    @staticmethod
    def write_fingerprints(preprocessed, smiles_data, offset, progress):
        for i in range(len(smiles_data)):
            smiles = smiles_data[i].decode('utf-8')
            molecule = Chem.MolFromSmiles(smiles)
            fingerprint = MACCSkeys.GenMACCSKeys(molecule)
            preprocessed[i + offset, :] = list(fingerprint)[1:]
            progress.increment()
