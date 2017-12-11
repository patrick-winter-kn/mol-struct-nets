from util import file_structure, thread_pool, file_util, progressbar, misc, concurrent_set, logger, constants, hdf5_util
import h5py
from steps.datageneration.randomsmiles import smiles_generator

number_threads = 1


class RandomSmiles:

    @staticmethod
    def get_id():
        return 'random_smiles'

    @staticmethod
    def get_name():
        return 'Random SMILES'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'n', 'name': 'Number of molecules', 'type': int, 'min': 1,
                           'description': 'Number of generated SMILES strings.'})
        parameters.append({'id': 'max_length', 'name': 'Maximum length', 'type': int, 'min': 1,
                           'description': 'Maximum length of a single SMILES string.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        pass

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(global_parameters, [constants.GlobalParameters.seed])
        hash_parameters.update(misc.copy_dict_from_keys(local_parameters, ['n', 'max_length']))
        file_name = str(local_parameters['n']) + 'x' + str(local_parameters['max_length']) + '_'\
            + misc.hash_parameters(hash_parameters) + '.h5'
        return file_util.resolve_subpath(file_structure.get_data_set_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        data_set_path = RandomSmiles.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.data_set] = file_util.get_filename(data_set_path, False)
        if file_util.file_exists(data_set_path):
            logger.log('Skipping step: ' + data_set_path + ' already exists')
        else:
            global_parameters[constants.GlobalParameters.n] = local_parameters['n']
            temp_data_set_path = file_util.get_temporary_file_path('random_smiles_data')
            data_h5 = h5py.File(temp_data_set_path, 'w')
            smiles_data = hdf5_util.create_dataset(data_h5, file_structure.DataSet.smiles, (local_parameters['n'],),
                                                   'S' + str(local_parameters['max_length']))
            chunks = misc.chunk(local_parameters['n'], number_threads)
            smiles_set = concurrent_set.ConcurrentSet()
            with progressbar.ProgressBar(local_parameters['n']) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        generator = smiles_generator\
                            .SmilesGenerator(chunk['size'], local_parameters['max_length'],
                                             global_parameters[constants.GlobalParameters.seed], chunk['start'],
                                             progress, smiles_set)
                        pool.submit(generator.write_smiles, smiles_data)
                    pool.wait()
            data_h5.close()
            file_util.move_file(temp_data_set_path, data_set_path)
