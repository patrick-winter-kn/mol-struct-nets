from util import data_validation, misc, file_structure, file_util, logger, progressbar, concurrent_max, concurrent_set,\
    thread_pool
import numpy
import h5py


number_threads = thread_pool.default_number_threads


class SmilesMatrix:

    @staticmethod
    def get_id():
        return 'smiles_matrix'

    @staticmethod
    def get_name():
        return 'SMILES Matrix'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'max_length', 'name': 'Maximum length (default: automatic)', 'type': int,
                           'default': None})
        parameters.append({'id': 'characters', 'name': 'Force characters (default: none)', 'type': str,
                           'default': None})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, parameters):
        hash_parameters = misc.copy_dict_from_keys(parameters, ['max_length', 'characters'])
        file_name = 'smiles_matrix_' + misc.hash_parameters(hash_parameters) + '.h5'
        return file_util.resolve_subpath(file_structure.get_preprocessed_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, parameters):
        preprocessed_path = SmilesMatrix.get_result_file(global_parameters, parameters)
        global_parameters['preprocessed_data'] = preprocessed_path
        if file_util.file_exists(preprocessed_path):
            logger.log('Skipping step: ' + preprocessed_path + ' already exists')
            preprocessed_h5 = h5py.File(preprocessed_path, 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            global_parameters['input_dimensions'] = (preprocessed.shape[1], preprocessed.shape[2])
            preprocessed_h5.close()
        else:
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles_data = data_h5[file_structure.DataSet.smiles]
            temp_preprocessed_path = file_util.get_temporary_file_path('smiles_matrix')
            preprocessed_h5 = h5py.File(temp_preprocessed_path, 'w')
            chunks = misc.chunk(len(smiles_data), number_threads)
            characters = concurrent_set.ConcurrentSet()
            max_length = concurrent_max.ConcurrentMax()
            print('Analyzing SMILES')
            with progressbar.ProgressBar(len(smiles_data)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(SmilesMatrix.analyze_smiles, smiles_data[chunk['start']:chunk['end'] + 1],
                                    characters, max_length, progress)
                    pool.wait()
            characters = sorted(characters.get_set_copy())
            index = preprocessed_h5.create_dataset('index', (len(characters),), dtype='S1')
            index_lookup = {}
            for i in range(len(characters)):
                index_lookup[characters[i]] = i
                index[i] = characters[i].encode('utf-8')
            global_parameters['input_dimensions'] = (max_length.get_max(), len(index))
            preprocessed = preprocessed_h5.create_dataset(file_structure.Preprocessed.preprocessed,
                                                          (len(smiles_data), max_length.get_max(), len(index)),
                                                          dtype='I')
            print('Writing matrices')
            with progressbar.ProgressBar(len(smiles_data)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(SmilesMatrix.write_smiles_matrices, preprocessed,
                                    smiles_data[chunk['start']:chunk['end'] + 1], index_lookup, max_length.get_max(),
                                    chunk['start'], progress)
                    pool.wait()
            data_h5.close()
            preprocessed_h5.close()
            file_util.move_file(temp_preprocessed_path, preprocessed_path)

    @staticmethod
    def analyze_smiles(smiles_data, characters, max_length, progress):
        characters.add(' ')
        for string in smiles_data:
            string = string.decode('utf-8')
            max_length.add_value(len(string))
            for character in string:
                characters.add(character)
            progress.increment()

    @staticmethod
    def pad_string(string, length):
        return string + (' ' * (length - len(string)))

    @staticmethod
    def string_to_matrix(string, index_lookup):
        matrix = numpy.zeros((len(string), len(index_lookup)))
        for i in range(len(string)):
            matrix[i][index_lookup[string[i]]] = 1
        return matrix

    @staticmethod
    def write_smiles_matrices(preprocessed, smiles_data, index_lookup, max_length, offset, progress):
        for i in range(len(smiles_data)):
            string = SmilesMatrix.pad_string(smiles_data[i].decode('utf-8'), max_length)
            preprocessed[i + offset] = SmilesMatrix.string_to_matrix(string, index_lookup)
            progress.increment()
