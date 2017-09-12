from util import data_validation, file_structure, file_util, misc, progressbar, thread_pool, logger, constants,\
    hdf5_util
from rdkit import Chem
import h5py


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
        parameters.append({'id': 'substructures', 'name': 'Substructures (separated by ;)', 'type': str,
                           'description': 'Substructures in SMILES format, separated by a semicolon.'})
        parameters.append({'id': 'logic', 'name': 'Logic expression (e.g. a&(b|c))', 'type': str, 'default': None,
                           'description': 'If this logic expression is true, the molecule will be active. a will be'
                                          ' true if the first substructure was found in the molecule. Default behaviour'
                                          ' is a&b&c&... .'})
        parameters.append({'id': 'name', 'name': 'Target name (default: None)', 'type': str, 'default': None,
                           'description': 'Prefix to the filename of the generated target data set.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(local_parameters, ['substructures', 'logic'])
        file_name = misc.hash_parameters(hash_parameters) + '.h5'
        if local_parameters['name'] is not None:
            file_name = local_parameters['name'] + '_' + file_name
        return file_util.resolve_subpath(file_structure.get_target_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        target_path = Substructure.get_result_file(global_parameters, local_parameters)
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
            with progressbar.ProgressBar(len(smiles_data)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(Substructure._generate_activities,
                                    smiles_data[chunk['start']:chunk['end'] + 1], substructures, logic, classes,
                                    chunk['start'], progress)
                    pool.wait()
            data_h5.close()
            target_h5.close()
            hdf5_util.set_property(temp_target_path, 'substructures', local_parameters['substructures'])
            hdf5_util.set_property(temp_target_path, 'logic', logic)
            file_util.move_file(temp_target_path, target_path)

    @staticmethod
    def _generate_activities(smiles_data, substructures, logic, classes, offset, progress):
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
            if eval(expression):
                classes[i + offset, 0] = 1.0
                classes[i + offset, 1] = 0.0
            else:
                classes[i + offset, 0] = 0.0
                classes[i + offset, 1] = 1.0
            progress.increment()
