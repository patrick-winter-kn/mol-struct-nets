from util import data_validation, file_structure, file_util, misc, multithread_progress, thread_pool
from rdkit import Chem
import h5py
# TODO hash at the end of target data set


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
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def execute(global_parameters, parameters):
        target_path = file_structure.get_target_file(global_parameters)
        if file_util.file_exists(target_path):
            print('Skipping step: ' + target_path + ' already exists')
        else:
            substructures = []
            for string in parameters['substructures'].split(';'):
                substructures.append(Chem.MolFromSmiles(string, sanitize=False))
            data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
            smiles_data = data_h5['smiles']
            temp_target_path = file_util.get_temporary_file_path('substructure_target_data')
            target_h5 = h5py.File(temp_target_path, 'w')
            classes = target_h5.create_dataset('classes', (smiles_data.shape[0], 2))
            logic = parameters['logic']
            if logic is None:
                logic = 'a'
                for i in range(1, len(substructures)):
                    logic += '&' + chr(ord('a')+i)
            chunks = misc.chunk(len(smiles_data), number_threads)
            with multithread_progress.MultithreadProgress(len(smiles_data)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(Substructure._generate_activities,
                                    smiles_data[chunk['start']:chunk['end'] + 1], substructures, logic, classes,
                                    chunk['start'], progress)
                    pool.wait()
            data_h5.close()
            target_h5.close()
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
