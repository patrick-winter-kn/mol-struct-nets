from util import file_structure, thread_pool, file_util, multithread_progress, misc, duplicate_checker
import h5py
from steps.datageneration.randomsmiles import smiles_generator
# TODO hash at the end of data set name

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
        parameters.append({'id': 'n', 'name': 'Number of molecules', 'type': int,
                           'description': 'Number of generated SMILES strings.'})
        parameters.append({'id': 'max_length', 'name': 'Maximum length', 'type': int,
                           'description': 'Maximum length of a single SMILES string.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        pass

    @staticmethod
    def execute(global_parameters, parameters):
        data_set_path = file_structure.get_data_set_file(global_parameters)
        if file_util.file_exists(data_set_path):
            print('Skipping step: ' + data_set_path + ' already exists')
        else:
            temp_data_set_path = file_util.get_temporary_file_path('random_smiles_data')
            data_h5 = h5py.File(temp_data_set_path, 'w')
            smiles_data = data_h5.create_dataset(file_structure.DataSet.smiles, (parameters['n'],),
                                                 'S' + str(parameters['max_length']))
            chunks = misc.chunk(parameters['n'], number_threads)
            checker = duplicate_checker.DuplicateChecker()
            with multithread_progress.MultithreadProgress(parameters['n']) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        generator = smiles_generator.SmilesGenerator(chunk['size'], parameters['max_length'],
                                                                     global_parameters['seed'], chunk['start'],
                                                                     progress, checker)
                        pool.submit(generator.write_smiles, smiles_data)
                    pool.wait()
            data_h5.close()
            file_util.move_file(temp_data_set_path, data_set_path)
