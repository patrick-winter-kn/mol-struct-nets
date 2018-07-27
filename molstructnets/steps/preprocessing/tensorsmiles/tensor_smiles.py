import h5py
import numpy

from util import data_validation, misc, file_structure, file_util, logger, progressbar, concurrent_max, concurrent_set, \
    thread_pool, constants, hdf5_util

number_threads = thread_pool.default_number_threads


class TensorSmiles:

    @staticmethod
    def get_id():
        return 'tensor_smiles'

    @staticmethod
    def get_name():
        return 'SMILES Tensor'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'min_length', 'name': 'Minimum Length', 'type': int, 'default': 1, 'min': 1,
                           'description': 'Minimum number of characters of input SMILES string. If a SMILES string is'
                                          ' below this value it will be padded with spaces. This value can overwrite'
                                          ' the Maximum Length parameter. Default: 1'})
        parameters.append({'id': 'max_length', 'name': 'Maximum Length', 'type': int, 'default': None, 'min': 1,
                           'description': 'Maximum number of characters of input SMILES string. If the limit is'
                                          ' exceeded the string will be shortened to fit into the tensor. Default:'
                                          ' automatic'})
        parameters.append({'id': 'characters', 'name': 'Force Characters', 'type': str, 'default': None,
                           'description': 'Characters in the given string will be added to the index in addition to'
                                          ' characters found in the data set. Default: None'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(local_parameters, ['min_length', 'max_length', 'characters'])
        file_name = 'tensor_smiles_' + misc.hash_parameters(hash_parameters) + '.h5'
        return file_util.resolve_subpath(file_structure.get_preprocessed_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        preprocessed_path = TensorSmiles.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.preprocessed_data] = preprocessed_path
        if file_util.file_exists(preprocessed_path):
            logger.log('Skipping step: ' + preprocessed_path + ' already exists')
            preprocessed_h5 = h5py.File(preprocessed_path, 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            global_parameters[constants.GlobalParameters.input_dimensions] = (preprocessed.shape[1],
                                                                              preprocessed.shape[2])
            preprocessed_h5.close()
        else:
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles_data = data_h5[file_structure.DataSet.smiles]
            temp_preprocessed_path = file_util.get_temporary_file_path('tensor_smiles')
            preprocessed_h5 = h5py.File(temp_preprocessed_path, 'w')
            chunks = misc.chunk(len(smiles_data), number_threads)
            characters = concurrent_set.ConcurrentSet()
            if local_parameters['characters'] is not None:
                for character in local_parameters['characters']:
                    characters.add(character)
            max_length = concurrent_max.ConcurrentMax()
            if local_parameters['max_length'] is not None:
                max_length.add(local_parameters['max_length'])
            logger.log('Analyzing SMILES')
            with progressbar.ProgressBar(len(smiles_data)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(TensorSmiles.analyze_smiles, smiles_data[chunk['start']:chunk['end']],
                                    characters, max_length, progress)
                    pool.wait()
            characters = sorted(characters.get_set_copy())
            index = hdf5_util.create_dataset(preprocessed_h5, 'index', (len(characters),), dtype='S1')
            index_lookup = {}
            for i in range(len(characters)):
                index_lookup[characters[i]] = i
                index[i] = characters[i].encode('utf-8')
            length = max(max_length.get_max(), local_parameters['min_length'])
            global_parameters[constants.GlobalParameters.input_dimensions] = (length, len(index))
            preprocessed = hdf5_util.create_dataset(preprocessed_h5, file_structure.Preprocessed.preprocessed,
                                                    (len(smiles_data), length, len(index)), dtype='I',
                                                    chunks=(1, length, len(index)))
            logger.log('Writing tensors')
            with progressbar.ProgressBar(len(smiles_data)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(TensorSmiles.write_smiles_tensors, preprocessed,
                                    smiles_data[chunk['start']:chunk['end']], index_lookup, length, chunk['start'],
                                    progress)
                    pool.wait()
            data_h5.close()
            preprocessed_h5.close()
            file_util.move_file(temp_preprocessed_path, preprocessed_path)

    @staticmethod
    def analyze_smiles(smiles_data, characters, max_length, progress):
        characters.add(' ')
        for string in smiles_data:
            string = string.decode('utf-8')
            max_length.add(len(string))
            for character in string:
                characters.add(character)
            progress.increment()

    @staticmethod
    def pad_string(string, length):
        return string + (' ' * (length - len(string)))

    @staticmethod
    def string_to_tensor(string, index_lookup):
        tensor = numpy.zeros((len(string), len(index_lookup)))
        for i in range(len(string)):
            tensor[i][index_lookup[string[i]]] = 1
        return tensor

    @staticmethod
    def write_smiles_tensors(preprocessed, smiles_data, index_lookup, max_length, offset, progress):
        for i in range(len(smiles_data)):
            string = TensorSmiles.pad_string(smiles_data[i].decode('utf-8'), max_length)
            preprocessed[i + offset] = TensorSmiles.string_to_tensor(string, index_lookup)
            progress.increment()
