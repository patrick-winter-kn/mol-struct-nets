import h5py

from steps.interpretation.shared import tensor_2d_renderer
from steps.interpretation.shared.kerasviz import cam
from util import data_validation, file_structure, file_util, progressbar, logger, misc, thread_pool, constants
from steps.preprocessing.shared.tensor2d import tensor_2d_jit_array


number_threads = thread_pool.default_number_threads


class RenderCams2DJit:

    @staticmethod
    def get_id():
        return 'render_cams_2d_jit'

    @staticmethod
    def get_name():
        return 'Render CAMs 2D JIT'

    @staticmethod
    def get_parameters():
        return list()

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_preprocessed_jit(global_parameters)
        data_validation.validate_cam(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        cam_h5 = h5py.File(file_structure.get_cam_file(global_parameters), 'r')
        preprocessed_h5 = h5py.File(global_parameters[constants.GlobalParameters.preprocessed_data], 'r')
        preprocessed = tensor_2d_jit_array.load_array(global_parameters)
        symbols = preprocessed_h5[file_structure.PreprocessedTensor2DJit.symbols][:]
        if file_structure.Cam.cam_active in cam_h5.keys():
            active_dir_path = file_util.resolve_subpath(file_structure.get_interpretation_folder(global_parameters),
                                                        'rendered_cams_active')
            file_util.make_folders(active_dir_path, True)
            cam_active = cam_h5[file_structure.Cam.cam_active]
            if file_structure.Cam.cam_active_indices in cam_h5.keys():
                indices = cam_h5[file_structure.Cam.cam_active_indices][:]
            else:
                indices = range(len(cam_active))
            logger.log('Rendering active CAMs', logger.LogLevel.INFO)
            chunks = misc.chunk(len(preprocessed), number_threads)
            with progressbar.ProgressBar(len(indices)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(RenderCams2DJit.render, preprocessed, cam_active, indices,
                                    symbols, active_dir_path, chunk['start'], chunk['end'], progress)
                    pool.wait()
        if file_structure.Cam.cam_inactive in cam_h5.keys():
            inactive_dir_path = file_util.resolve_subpath(file_structure.get_interpretation_folder(global_parameters),
                                                          'rendered_cams_inactive')
            file_util.make_folders(inactive_dir_path, True)
            cam_inactive = cam_h5[file_structure.Cam.cam_inactive]
            if file_structure.Cam.cam_inactive_indices in cam_h5.keys():
                indices = cam_h5[file_structure.Cam.cam_inactive_indices][:]
            else:
                indices = range(cam_inactive)
            logger.log('Rendering inactive CAMs', logger.LogLevel.INFO)
            chunks = misc.chunk(len(preprocessed), number_threads)
            with progressbar.ProgressBar(len(indices)) as progress:
                with thread_pool.ThreadPool(number_threads) as pool:
                    for chunk in chunks:
                        pool.submit(RenderCams2DJit.render, preprocessed, cam_inactive, indices,
                                    symbols, inactive_dir_path, chunk['start'], chunk['end'], progress)
                    pool.wait()
        cam_h5.close()
        preprocessed_h5.close()

    @staticmethod
    def render(preprocessed, data_set, indices, symbols, output_dir_path, start, end, progress):
            for i in indices[start:end+1]:
                output_path = file_util.resolve_subpath(output_dir_path, str(i) + '.svgz')
                if not file_util.file_exists(output_path):
                    heatmap = cam.array_to_heatmap([data_set[i][:]])
                    tensor_2d_renderer.render(output_path, preprocessed[i], symbols, heatmap=heatmap)
                progress.increment()
