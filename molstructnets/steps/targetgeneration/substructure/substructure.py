from util import data_validation, file_structure, file_util, misc, progressbar, thread_pool, logger, constants,\
    hdf5_util
from rdkit import Chem
import h5py
import random


number_threads = 1


class Substructure:

    @staticmethod
    def get_id():
        return 'substructure'

    @staticmethod
    def get_name():
        return 'Substructure'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'substructures', 'name': 'Substructures', 'type': str,
                           'description': 'Substructures in SMILES format, separated by a semicolon.'})
        parameters.append({'id': 'logic', 'name': 'Logic Expression', 'type': str, 'default': None,
                           'description': 'If this logic expression is true, the molecule will be active. a will be'
                                          ' true if the first substructure was found in the molecule. Example: a&(b|c).'
                                          ' Default behaviour is a&b&c&... .'})
        parameters.append({'id': 'name', 'name': 'Target Name', 'type': str, 'default': None,
                           'description': 'Prefix to the filename of the generated target data set. Default: same as'
                                          ' substructures'})
        parameters.append({'id': 'error', 'name': 'Error probability', 'type': int, 'default': 0,
                           'description': 'Probability in percent that a molecule is assigned to the wrong class.'
                                          ' Default: 0%'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(local_parameters, ['substructures', 'logic', 'error'])
        file_name = misc.hash_parameters(hash_parameters) + '.h5'
        if local_parameters['name'] is not None:
            substructure_name = local_parameters['name']
        else:
            substructure_name = local_parameters['substructures']
        substructure_name = substructure_name.replace('/', '')
        file_name = substructure_name + '_' + file_name
        return file_util.resolve_subpath(file_structure.get_target_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        target_path = Substructure.get_result_file(global_parameters, local_parameters)
        if constants.GlobalParameters.target in global_parameters:
            logger.log('Target has already been specified. Overwriting target parameter with generated target.',
                       logger.LogLevel.WARNING)
        global_parameters[constants.GlobalParameters.target] = file_util.get_filename(target_path, False)
        if file_util.file_exists(target_path):
            logger.log('Skipping step: ' + target_path + ' already exists')
        else:
            substructures = []
            for string in local_parameters['substructures'].split(';'):
                substructures.append(Chem.MolFromSmiles(string, sanitize=False))
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles_data = data_h5[file_structure.DataSet.smiles]
            temp_target_path = file_util.get_temporary_file_path('substructure_target_data')
            target_h5 = h5py.File(temp_target_path, 'w')
            classes = hdf5_util.create_dataset(target_h5, file_structure.Target.classes, (smiles_data.shape[0], 2))
            logic = local_parameters['logic']
            if logic is None:
                logic = 'a'
                for i in range(1, len(substructures)):
                    logic += '&' + chr(ord('a')+i)
            chunks = misc.chunk(len(smiles_data), number_threads)
            error = local_parameters['error'] * 0.01
            with progressbar.ProgressBar(len(smiles_data)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for i in range(len(chunks)):
                        chunk = chunks[i]
                        pool.submit(Substructure._generate_activities,
                                    smiles_data[chunk['start']:chunk['end'] + 1], substructures, logic, classes,
                                    chunk['start'], error, global_parameters[constants.GlobalParameters.seed] + i,
                                    progress)
                    pool.wait()
            data_h5.close()
            target_h5.close()
            hdf5_util.set_property(temp_target_path, 'substructures', local_parameters['substructures'])
            hdf5_util.set_property(temp_target_path, 'logic', logic)
            file_util.move_file(temp_target_path, target_path)

    @staticmethod
    def _generate_activities(smiles_data, substructures, logic, classes, offset, error, random_seed, progress):
        random_ = random.Random(random_seed)
        for i in range(len(smiles_data)):
            structure = Chem.MolFromSmiles(smiles_data[i].decode('utf-8'), sanitize=False)
            evals = []
            for substructure in substructures:
                evals.append(structure.HasSubstructMatch(substructure))
            expression = ''
            for character in logic:
                if misc.in_range(ord(character), ord('a'), ord('z')):
                    index = ord(character) - ord('a')
                    expression += str(evals[index])
                else:
                    expression += character
            active = eval(expression)
            if random_.random() < error:
                active = not active
            if active:
                classes[i + offset, 0] = 1.0
                classes[i + offset, 1] = 0.0
            else:
                classes[i + offset, 0] = 0.0
                classes[i + offset, 1] = 1.0
            progress.increment()
