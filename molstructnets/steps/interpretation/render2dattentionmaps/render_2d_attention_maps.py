import h5py

from steps.interpretation.shared import matrix_2d_renderer
from steps.interpretation.shared.kerasviz import attention_map
from util import data_validation, file_structure, file_util, progressbar, logger, misc, thread_pool, constants


number_threads = thread_pool.default_number_threads


class Render2DAttentionMaps:

    @staticmethod
    def get_id():
        return 'render_2d_attention_maps'

    @staticmethod
    def get_name():
        return 'Render 2D Attention Maps'

    @staticmethod
    def get_parameters():
        parameters = list()
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_preprocessed(global_parameters)
        data_validation.validate_attention_map(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        attention_map_h5 = h5py.File(file_structure.get_attentionmap_file(global_parameters), 'r')
        preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data], 'r')
        preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
        symbols = preprocessed_h5[file_structure.Preprocessed.index]
        if file_structure.AttentionMap.attention_map_active in attention_map_h5.keys():
            active_dir_path = file_util.resolve_subpath(file_structure.get_interpretation_folder(global_parameters),
                                                        'rendered_attention_active')
            file_util.make_folders(active_dir_path, True)
            attention_map_active = attention_map_h5[file_structure.AttentionMap.attention_map_active]
            if file_structure.AttentionMap.attention_map_active_indices in attention_map_h5.keys():
                indices = attention_map_h5[file_structure.AttentionMap.attention_map_active_indices]
            else:
                indices = range(len(attention_map_active))
            logger.log('Rendering active attention maps', logger.LogLevel.INFO)
            chunks = misc.chunk(len(preprocessed), number_threads)
            with progressbar.ProgressBar(len(indices)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(Render2DAttentionMaps.render, preprocessed, attention_map_active, indices, symbols,
                                    active_dir_path, chunk['start'], chunk['end'], progress)
                    pool.wait()
        if file_structure.AttentionMap.attention_map_inactive in attention_map_h5.keys():
            inactive_dir_path = file_util.resolve_subpath(file_structure.get_interpretation_folder(global_parameters),
                                                          'rendered_attention_inactive')
            file_util.make_folders(inactive_dir_path, True)
            attention_map_inactive = attention_map_h5[file_structure.AttentionMap.attention_map_inactive]
            if file_structure.AttentionMap.attention_map_inactive_indices in attention_map_h5.keys():
                indices = attention_map_h5[file_structure.AttentionMap.attention_map_inactive_indices]
            else:
                indices = range(attention_map_inactive)
            logger.log('Rendering inactive attention maps', logger.LogLevel.INFO)
            chunks = misc.chunk(len(preprocessed), number_threads)
            with progressbar.ProgressBar(len(indices)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(Render2DAttentionMaps.render, preprocessed, attention_map_inactive, indices, symbols,
                                    inactive_dir_path, chunk['start'], chunk['end'], progress)
                    pool.wait()
        attention_map_h5.close()
        preprocessed_h5.close()

    @staticmethod
    def render(preprocessed, data_set, indices, symbols, output_dir_path, start, end, progress):
            for i in indices[start:end+1]:
                output_path = file_util.resolve_subpath(output_dir_path, str(i) + '.svg')
                if not file_util.file_exists(output_path):
                    heatmap = attention_map.array_to_heatmap([data_set[i]])
                    matrix_2d_renderer.render(output_path, preprocessed[i], symbols, heatmap=heatmap)
                progress.increment()
