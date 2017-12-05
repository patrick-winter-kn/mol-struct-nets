import h5py
import numpy
from steps.interpretation.shared import matrix_2d_renderer, smiles_renderer
from util import data_validation, file_structure, file_util, progressbar, logger, misc, thread_pool, constants


number_threads = thread_pool.default_number_threads


class RenderSubstructureAtoms:

    rgb_black = [0, 0, 0]
    rgb_red = [255, 0, 0]

    @staticmethod
    def get_id():
        return 'render_substructure_atoms'

    @staticmethod
    def get_name():
        return 'Render Substructure Atoms'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id': 'renderer', 'name': 'Renderer (smiles or 2d, default: automatic)', 'type': str,
                           'default': None,
                           'description': 'The renderer that should be used. If automatic the smiles renderer is used '
                                          'for 1d data and the 2d renderer is used for 2d data.'})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_preprocessed(global_parameters)
        data_validation.validate_attention_map(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        attention_map_h5 = h5py.File(file_structure.get_attentionmap_file(global_parameters), 'r')
        data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
        smiles = data_h5[file_structure.DataSet.smiles]
        preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data], 'r')
        preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
        symbols = preprocessed_h5[file_structure.Preprocessed.index]
        renderer = local_parameters['renderer']
        if renderer is None:
            if len(global_parameters[constants.GlobalParameters.input_dimensions]) == 2:
                renderer = 'smiles'
            elif len(global_parameters[constants.GlobalParameters.input_dimensions]) == 3:
                renderer = '2d'
            else:
                raise ValueError('Unsupported dimensionality for rendering')
        active_dir_path = file_util.resolve_subpath(file_structure.get_interpretation_folder(global_parameters),
                                                    'rendered_substructure_atoms')
        file_util.make_folders(active_dir_path, True)
        substructure_atoms = attention_map_h5[file_structure.AttentionMap.substructure_atoms]
        indices = range(len(substructure_atoms))
        logger.log('Rendering substructure atoms', logger.LogLevel.INFO)
        chunks = misc.chunk(len(preprocessed), number_threads)
        with progressbar.ProgressBar(len(indices)) as progress:
            with thread_pool.ThreadPool(number_threads) as pool:
                for chunk in chunks:
                    pool.submit(RenderSubstructureAtoms.render, preprocessed, substructure_atoms, indices, smiles,
                                symbols, active_dir_path, renderer, chunk['start'], chunk['end'], progress)
                pool.wait()
        attention_map_h5.close()
        preprocessed_h5.close()
        data_h5.close()

    @staticmethod
    def render(preprocessed, data_set, indices, smiles, symbols, output_dir_path, renderer, start, end, progress):
            for i in indices[start:end+1]:
                output_path = file_util.resolve_subpath(output_dir_path, str(i) + '.svg')
                if not file_util.file_exists(output_path):
                    smiles_string = smiles[i].decode('utf-8')
                    heatmap = RenderSubstructureAtoms.generate_heatmap(data_set[i])
                    if renderer == 'smiles':
                        smiles_renderer.render(smiles_string, output_path, 5, heatmap)
                    elif renderer == '2d':
                        matrix_2d_renderer.render(output_path, preprocessed[i], symbols, heatmap=heatmap)
                progress.increment()

    @staticmethod
    def generate_heatmap(substructure_atoms):
        shape = list(substructure_atoms.shape)
        shape.append(3)
        shape = tuple(shape)
        heatmap = numpy.zeros(shape, dtype='uint8')
        RenderSubstructureAtoms.assign_color(substructure_atoms, heatmap)
        return heatmap

    @staticmethod
    def assign_color(substructure_atoms, heatmap):
        if isinstance(substructure_atoms, numpy.ndarray):
            for i in range(len(substructure_atoms)):
                RenderSubstructureAtoms.assign_color(substructure_atoms[i], heatmap[i])
        else:
            if substructure_atoms == 1:
                heatmap[:] = RenderSubstructureAtoms.rgb_red[:]
            else:
                heatmap[:] = RenderSubstructureAtoms.rgb_black[:]
