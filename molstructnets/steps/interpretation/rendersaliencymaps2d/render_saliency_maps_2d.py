import h5py

from steps.interpretation.shared import tensor_2d_renderer
from steps.interpretation.shared.kerasviz import saliency_map
from steps.preprocessing.shared.tensor2d import tensor_2d_array
from util import data_validation, file_structure, file_util, logger, constants, multi_process_progressbar,\
    process_pool, manager


class RenderSaliencyMaps2D:

    @staticmethod
    def get_id():
        return 'render_saliency_maps_2d'

    @staticmethod
    def get_name():
        return 'Render Saliency Maps 2D'

    @staticmethod
    def get_parameters():
        return list()

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_preprocessed_specs(global_parameters)
        data_validation.validate_saliency_map(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        saliency_map_h5 = h5py.File(file_structure.get_saliency_map_file(global_parameters), 'r')
        preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data], 'r')
        symbols = preprocessed_h5[file_structure.PreprocessedTensor2D.symbols][:]
        preprocessed_h5.close()
        if file_structure.SaliencyMap.saliency_map_active in saliency_map_h5.keys():
            active_dir_path = file_util.resolve_subpath(file_structure.get_interpretation_folder(global_parameters),
                                                        'rendered_saliency_maps_active')
            file_util.make_folders(active_dir_path, True)
            saliency_map_active = saliency_map_h5[file_structure.SaliencyMap.saliency_map_active]
            if file_structure.SaliencyMap.saliency_map_active_indices in saliency_map_h5.keys():
                indices = saliency_map_h5[file_structure.SaliencyMap.saliency_map_active_indices][:]
            else:
                indices = range(len(saliency_map_active))
            logger.log('Rendering active saliency maps', logger.LogLevel.INFO)
            queue = manager.instance.Queue(10)
            with multi_process_progressbar.MultiProcessProgressbar(len(indices), value_buffer=10) as progress:
                with process_pool.ProcessPool(process_pool.default_number_processes) as pool:
                    for i in range(process_pool.default_number_processes):
                        pool.submit(render, global_parameters, symbols, active_dir_path, queue, progress.get_slave())
                    for i in indices:
                        queue.put((i, saliency_map_active[i][:]))
                    for i in range(process_pool.default_number_processes):
                        queue.put(None)
                    pool.wait()
        if file_structure.SaliencyMap.saliency_map_inactive in saliency_map_h5.keys():
            inactive_dir_path = file_util.resolve_subpath(file_structure.get_interpretation_folder(global_parameters),
                                                          'rendered_saliency_maps_inactive')
            file_util.make_folders(inactive_dir_path, True)
            saliency_map_inactive = saliency_map_h5[file_structure.SaliencyMap.saliency_map_inactive]
            if file_structure.SaliencyMap.saliency_map_inactive_indices in saliency_map_h5.keys():
                indices = saliency_map_h5[file_structure.SaliencyMap.saliency_map_inactive_indices][:]
            else:
                indices = range(saliency_map_inactive)
            logger.log('Rendering inactive saliency maps', logger.LogLevel.INFO)
            queue = manager.instance.Queue(10)
            with multi_process_progressbar.MultiProcessProgressbar(len(indices), value_buffer=10) as progress:
                with process_pool.ProcessPool(process_pool.default_number_processes) as pool:
                    for i in range(process_pool.default_number_processes):
                        pool.submit(render, global_parameters, symbols, inactive_dir_path, queue, progress.get_slave())
                    for i in indices:
                        queue.put((i, saliency_map_inactive[i][:]))
                    for i in range(process_pool.default_number_processes):
                        queue.put(None)
                    pool.wait()
        saliency_map_h5.close()


def render(global_parameters, symbols, output_dir_path, queue, progress):
    preprocessed = tensor_2d_array.load_array(global_parameters, multi_process=False)
    while True:
        next = queue.get()
        if next is None:
            break
        index, data = next
        output_path = file_util.resolve_subpath(output_dir_path, str(index) + '.svgz')
        if not file_util.file_exists(output_path):
            heatmap = saliency_map.array_to_heatmap([data])
            tensor_2d_renderer.render(output_path, preprocessed[index], symbols, heatmap=heatmap)
        progress.increment()
    progress.finish()
    preprocessed.close()
