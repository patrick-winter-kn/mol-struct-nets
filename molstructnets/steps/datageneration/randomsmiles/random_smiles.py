from util import file_structure, process_pool, file_util, misc, logger, constants, hdf5_util, multi_process_progressbar
import h5py
from steps.datageneration.randomsmiles import smiles_generator


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
        parameters.append({'id': 'n', 'name': 'Number of Molecules', 'type': int, 'min': 1,
                           'description': 'Number of generated SMILES strings.'})
        parameters.append({'id': 'max_length', 'name': 'Maximum Length', 'type': int, 'min': 1,
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
        if constants.GlobalParameters.data_set in global_parameters:
            logger.log('Data set has already been specified. Overwriting data set parameter with generated data.',
                       logger.LogLevel.WARNING)
        global_parameters[constants.GlobalParameters.data_set] = file_util.get_filename(data_set_path, False)
        if file_util.file_exists(data_set_path):
            logger.log('Skipping step: ' + data_set_path + ' already exists')
        else:
            global_parameters[constants.GlobalParameters.n] = local_parameters['n']
            temp_data_set_path = file_util.get_temporary_file_path('random_smiles_data')
            chunks = misc.chunk(local_parameters['n'], process_pool.default_number_processes)
            with process_pool.ProcessPool(len(chunks)) as pool:
                with multi_process_progressbar.MultiProcessProgressbar(local_parameters['n'], value_buffer=100) as progress:
                    for chunk in chunks:
                        generator = smiles_generator\
                            .SmilesGenerator(chunk['size'], local_parameters['max_length'],
                                             global_parameters[constants.GlobalParameters.seed], chunk['start'])
                        pool.submit(generator.generate_smiles_batch, progress=progress.get_slave())
                    results = pool.get_results()
            data_h5 = h5py.File(temp_data_set_path, 'w')
            smiles_data = hdf5_util.create_dataset(data_h5, file_structure.DataSet.smiles, (local_parameters['n'],),
                                                   'S' + str(local_parameters['max_length']))
            offset = 0
            for result in results:
                smiles_data[offset:offset + len(result)] = result[:]
                offset += len(result)
            data_h5.close()
            file_util.move_file(temp_data_set_path, data_set_path)
