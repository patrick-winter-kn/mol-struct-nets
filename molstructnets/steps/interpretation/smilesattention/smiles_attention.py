from util import data_validation, file_structure, file_util, progressbar, misc, constants
from keras import models, activations
import h5py
from vis.utils import utils
from vis import visualization
from steps.interpretation.smilesattention import smiles_renderer


class SmilesAttention:

    @staticmethod
    def get_id():
        return 'smiles_attention'

    @staticmethod
    def get_name():
        return 'SMILES Attention'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'top_n', 'name': 'Top n (default: all)', 'type': int, 'default': None,
                           'description': 'An attention map for the n highest scored molecules will be generated.'})
        parameters.append({'id': 'actives', 'name': 'Active class (otherwise inactive, default: True)', 'type': bool,
                           'default': True,
                           'description': 'If true the attention map will show the attention for the active class. If'
                                          ' false it will be for the inactive class.'})
        parameters.append({'id': 'correct_predictions', 'name': 'Only correct predictions (default: False)',
                           'type': bool, 'default': False,
                           'description': 'If true only correct predictions will be considered.'})
        parameters.append({'id': 'test_data', 'name': 'Use test data (default: True)', 'type': bool, 'default': True,
                           'description': 'If true the attention maps will be generated for the test data. Otherwise'
                                          ' the train data will be used.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)
        data_validation.validate_partition(global_parameters)
        data_validation.validate_preprocessed(global_parameters)
        data_validation.validate_network(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        model = models.load_model(file_structure.get_network_file(global_parameters))
        out_layer_index = len(model.layers)-1
        model.layers[out_layer_index].activation = activations.linear
        model = utils.apply_modifications(model)
        data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
        smiles = data_h5[file_structure.DataSet.smiles]
        target_h5 = h5py.File(file_structure.get_target_file(global_parameters), 'r')
        classes = target_h5[file_structure.Target.classes]
        prediction_h5 = h5py.File(file_structure.get_prediction_file(global_parameters), 'r')
        predictions = prediction_h5[file_structure.Predictions.prediction]
        partition_h5 = h5py.File(global_parameters[constants.GlobalParameters.partition_data], 'r')
        if local_parameters['test_data']:
            references = partition_h5[file_structure.Partitions.test]
        else:
            references = partition_h5[file_structure.Partitions.train]
        preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data], 'r')
        preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
        indices = predictions[:, 0].argsort()[::-1]
        if local_parameters['top_n'] is None:
            count = len(indices)
        else:
            count = min(local_parameters['top_n'], len(indices))
        if local_parameters['actives']:
            class_index = 0
        else:
            class_index = 1
        output_dir_path = file_util.resolve_subpath(file_structure.get_interpretation_folder(global_parameters),
                                                    'smiles_attention')
        file_util.make_folders(output_dir_path, True)
        j = 0
        with progressbar.ProgressBar(count) as progress:
            for i in range(count):
                index = -1
                while index is not None and index not in references:
                    if j >= len(indices):
                        index = None
                    else:
                        index = indices[j]
                    if local_parameters['correct_predictions']:
                        if misc.is_active(classes[index]) is not misc.is_active(predictions[index]):
                            index = -1
                    j += 1
                if index is not None:
                    output_path = file_util.resolve_subpath(output_dir_path, str(index) + '.svg')
                    if not file_util.file_exists(output_path):
                        smiles_string = smiles[index].decode('utf-8')
                        smiles_matrix = preprocessed[index]
                        heatmap = visualization.visualize_saliency(model, out_layer_index, filter_indices=[class_index],
                                                                   seed_input=smiles_matrix)
                        smiles_renderer.render(smiles_string, output_path, 5, heatmap)
                progress.increment()
        data_h5.close()
        target_h5.close()
        prediction_h5.close()
        partition_h5.close()
        preprocessed_h5.close()
