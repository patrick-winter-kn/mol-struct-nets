import h5py
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy

from util import data_validation, misc, file_structure, file_util, logger, process_pool, constants,\
    hdf5_util, multi_process_progressbar


class EcfpFingerprint:

    @staticmethod
    def get_id():
        return 'ecfp_fingerprint'

    @staticmethod
    def get_name():
        return 'ECFP Fingerprint'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'radius', 'name': 'Radius', 'type': int, 'default': 2, 'min': 0,
                           'description': 'The radius of the fingerprint. Default: 2'})
        parameters.append({'id': 'nr_values', 'name': 'Number of values', 'type': int, 'default': 1024, 'min': 2,
                           'description': 'The number of values of the fingerprint. Default: 1024'})
        parameters.append({'id': 'count', 'name': 'Use counts', 'type': bool, 'default': False,
                           'description': 'Use counts instead of bits. Default: False'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(local_parameters, ['radius', 'nr_values', 'count'])
        file_name = 'ecfpfingerprint_' + misc.hash_parameters(hash_parameters) + '.h5'
        return file_util.resolve_subpath(file_structure.get_preprocessed_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        preprocessed_path = EcfpFingerprint.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.preprocessed_data] = preprocessed_path
        if file_util.file_exists(preprocessed_path):
            logger.log('Skipping step: ' + preprocessed_path + ' already exists')
            preprocessed_h5 = h5py.File(preprocessed_path, 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            global_parameters[constants.GlobalParameters.input_dimensions] = (preprocessed.shape[1],)
            preprocessed_h5.close()
        else:
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles_data = data_h5[file_structure.DataSet.smiles][:]
            data_h5.close()
            temp_preprocessed_path = file_util.get_temporary_file_path('ecfpfingerprint')
            chunks = misc.chunk(len(smiles_data), process_pool.default_number_processes)
            global_parameters[constants.GlobalParameters.input_dimensions] = (local_parameters['nr_values'],)
            logger.log('Calculating fingerprints')
            with process_pool.ProcessPool(len(chunks)) as pool:
                with multi_process_progressbar.MultiProcessProgressbar(len(smiles_data), value_buffer=100) as progress:
                    for chunk in chunks:
                        pool.submit(generate_fingerprints, smiles_data[chunk['start']:chunk['end'] + 1],
                                    local_parameters['radius'], local_parameters['nr_values'], local_parameters['count'],
                                    progress=progress.get_slave())
                    results = pool.get_results()
            dtype = 'uint8'
            if local_parameters['count']:
                dtype = 'uint16'
            preprocessed_h5 = h5py.File(temp_preprocessed_path, 'w')
            preprocessed = hdf5_util.create_dataset(preprocessed_h5, file_structure.Preprocessed.preprocessed,
                                                    (len(smiles_data), local_parameters['nr_values']), dtype=dtype,
                                                    chunks=(1, local_parameters['nr_values']))
            offset = 0
            for result in results:
                preprocessed[offset:offset + len(result)] = result[:]
                offset += len(result)
            preprocessed_h5.close()
            file_util.move_file(temp_preprocessed_path, preprocessed_path)


def generate_fingerprints(smiles_data, radius, nr_values, count, progress=None):
    dtype = 'uint8'
    if count:
        dtype = 'uint16'
    preprocessed = numpy.zeros((len(smiles_data),nr_values), dtype=dtype)
    for i in range(len(smiles_data)):
        smiles = smiles_data[i].decode('utf-8')
        molecule = Chem.MolFromSmiles(smiles)
        if count:
            elements = AllChem.GetMorganFingerprint(molecule, radius).GetNonzeroElements()
            fingerprint = numpy.zeros(nr_values)
            for element in elements:
                fingerprint[element % nr_values] += elements[element]
        else:
            fingerprint = numpy.array(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nr_values))
        preprocessed[i] = fingerprint[:]
        if progress is not None:
            progress.increment()
    if progress is not None:
        progress.finish()
    return preprocessed
