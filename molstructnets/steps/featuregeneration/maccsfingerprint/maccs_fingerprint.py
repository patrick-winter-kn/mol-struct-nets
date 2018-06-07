import h5py
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy

from util import data_validation, misc, file_structure, file_util, logger, process_pool, constants, hdf5_util,\
    multi_process_progressbar


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
        file_name = 'maccsfingerprint.h5'
        return file_util.resolve_subpath(file_structure.get_preprocessed_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        preprocessed_path = MaccsFingerprint.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.preprocessed_data] = preprocessed_path
        if file_util.file_exists(preprocessed_path):
            logger.log('Skipping step: ' + preprocessed_path + ' already exists')
            global_parameters[constants.GlobalParameters.input_dimensions] = (166,)
        else:
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles_data = data_h5[file_structure.DataSet.smiles][:]
            data_h5.close()
            temp_preprocessed_path = file_util.get_temporary_file_path('maccsfingerprint')
            chunks = misc.chunk(len(smiles_data), process_pool.default_number_processes)
            global_parameters[constants.GlobalParameters.input_dimensions] = (166,)
            logger.log('Calculating fingerprints')
            with process_pool.ProcessPool(len(chunks)) as pool:
                with multi_process_progressbar.MultiProcessProgressbar(len(smiles_data), value_buffer=100) as progress:
                    for chunk in chunks:
                        pool.submit(generate_fingerprints, smiles_data[chunk['start']:chunk['end'] + 1],
                                    progress=progress.get_slave())
                    results = pool.get_results()
            preprocessed_h5 = h5py.File(temp_preprocessed_path, 'w')
            preprocessed = hdf5_util.create_dataset(preprocessed_h5, file_structure.Preprocessed.preprocessed,
                                                    (len(smiles_data), 166), dtype='uint8', chunks=(1, 166))
            offset = 0
            for result in results:
                preprocessed[offset:offset + len(result)] = result[:]
                offset += len(result)
            preprocessed_h5.close()
            file_util.move_file(temp_preprocessed_path, preprocessed_path)


def generate_fingerprints(smiles_data, progress=None):
    preprocessed = numpy.zeros((len(smiles_data), 166), dtype='uint8')
    for i in range(len(smiles_data)):
        smiles = smiles_data[i].decode('utf-8')
        molecule = Chem.MolFromSmiles(smiles)
        fingerprint = MACCSkeys.GenMACCSKeys(molecule)
        preprocessed[i, :] = list(fingerprint)[1:]
        if progress is not None:
            progress.increment()
    if progress is not None:
        progress.finish()
    return preprocessed
