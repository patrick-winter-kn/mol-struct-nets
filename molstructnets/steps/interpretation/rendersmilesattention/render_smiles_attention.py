import h5py

from steps.interpretation.shared import smiles_renderer
from steps.interpretation.shared.kerasviz import attention_map
from util import data_validation, file_structure, file_util, progressbar, logger, misc, thread_pool


number_threads = thread_pool.default_number_threads


class RenderSmilesAttention:

    @staticmethod
    def get_id():
        return 'render_smiles_attention'

    @staticmethod
    def get_name():
        return 'Render SMILES Attention'

    @staticmethod
    def get_parameters():
        parameters = list()
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_attention_map(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        attention_map_h5 = h5py.File(file_structure.get_attentionmap_file(global_parameters), 'r')
        data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
        smiles = data_h5[file_structure.DataSet.smiles]
        if file_structure.AttentionMap.attention_map_active in attention_map_h5.keys():
            active_dir_path = file_util.resolve_subpath(file_structure.get_interpretation_folder(global_parameters),
                                                        'smiles_attention_active')
            file_util.make_folders(active_dir_path, True)
            attention_map_active = attention_map_h5[file_structure.AttentionMap.attention_map_active]
            indices = None
            if file_structure.AttentionMap.attention_map_active_indices in attention_map_h5.keys():
                indices = attention_map_h5[file_structure.AttentionMap.attention_map_active_indices]
            else:
                indices = range(len(attention_map_active))
            logger.log('Rendering active SMILES attention maps', logger.LogLevel.INFO)
            chunks = misc.chunk(len(smiles), number_threads)
            with progressbar.ProgressBar(len(indices)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(RenderSmilesAttention.render, attention_map_active, indices, smiles, active_dir_path, chunk['start'], chunk['end'], progress)
                    pool.wait()
        if file_structure.AttentionMap.attention_map_inactive in attention_map_h5.keys():
            inactive_dir_path = file_util.resolve_subpath(file_structure.get_interpretation_folder(global_parameters),
                                                        'smiles_attention_inactive')
            file_util.make_folders(inactive_dir_path, True)
            attention_map_inactive = attention_map_h5[file_structure.AttentionMap.attention_map_inactive]
            if file_structure.AttentionMap.attention_map_inactive_indices in attention_map_h5.keys():
                indices = attention_map_h5[file_structure.AttentionMap.attention_map_inactive_indices]
            else:
                indices = range(attention_map_inactive)
            logger.log('Rendering inactive SMILES attention maps', logger.LogLevel.INFO)
            chunks = misc.chunk(len(smiles), number_threads)
            with progressbar.ProgressBar(len(indices)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(RenderSmilesAttention.render, attention_map_inactive, indices, smiles, inactive_dir_path, chunk['start'], chunk['end'], progress)
                    pool.wait()
        attention_map_h5.close()
        data_h5.close()


    @staticmethod
    def render(data_set, indices, smiles, output_dir_path, start, end, progress):
            for i in indices[start:end+1]:
                output_path = file_util.resolve_subpath(output_dir_path, str(i) + '.svg')
                if not file_util.file_exists(output_path):
                    smiles_string = smiles[i].decode('utf-8')
                    heatmap = attention_map.array_to_heatmap([data_set[i]])
                    smiles_renderer.render(smiles_string, output_path, 5, heatmap)
                progress.increment()
