from util import data_validation, misc, file_structure, file_util, logger, progressbar, thread_pool, constants
import h5py
from steps.preprocessing.image import image_renderer


number_threads = thread_pool.default_number_threads


class Image:

    @staticmethod
    def get_id():
        return 'image'

    @staticmethod
    def get_name():
        return 'Image'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'size', 'name': 'Size in pixels (n×n)', 'type': int,
                           'description': 'Number of horizontal / vertical pixels.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        hash_parameters = misc.copy_dict_from_keys(local_parameters, ['size'])
        file_name = 'image_' + misc.hash_parameters(hash_parameters)
        return file_util.resolve_subpath(file_structure.get_preprocessed_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        global_parameters[constants.GlobalParameters.input_dimensions] = (local_parameters['size'],
                                                                          local_parameters['size'], 3)
        preprocess_path = Image.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.preprocessed_data] = preprocess_path
        file_util.make_folders(preprocess_path, True)
        data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
        smiles_data = data_h5[file_structure.DataSet.smiles]
        chunks = misc.chunk(len(smiles_data), number_threads)
        logger.log('Rendering images')
        with progressbar.ProgressBar(len(smiles_data)) as progress:
            with thread_pool.ThreadPool(number_threads) as pool:
                for chunk in chunks:
                    renderer = image_renderer.ImageRenderer(preprocess_path, progress, local_parameters['size'])
                    pool.submit(renderer.render, smiles_data[chunk['start']:chunk['end'] + 1], chunk['start'])
                pool.wait()
        data_h5.close()
