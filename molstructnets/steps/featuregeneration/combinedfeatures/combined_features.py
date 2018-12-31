import h5py

from util import progressbar, file_structure, file_util, logger, constants, hdf5_util


class CombinedFeatures:

    @staticmethod
    def get_id():
        return 'combined_features'

    @staticmethod
    def get_name():
        return 'Combined Features'

    @staticmethod
    def get_parameters():
        return list()

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        if len(global_parameters[constants.GlobalParameters.feature_files]) < 2:
            raise ValueError('Not enough features to combine.')

    @staticmethod
    def get_result_file(global_parameters, local_parameters):
        file_name = 'combined_features.h5'
        return file_util.resolve_subpath(file_structure.get_result_folder(global_parameters), file_name)

    @staticmethod
    def execute(global_parameters, local_parameters):
        global_parameters[constants.GlobalParameters.feature_id] = 'combined'
        preprocessed_path = CombinedFeatures.get_result_file(global_parameters, local_parameters)
        global_parameters[constants.GlobalParameters.preprocessed_data] = preprocessed_path
        if file_util.file_exists(preprocessed_path):
            logger.log('Skipping step: ' + preprocessed_path + ' already exists')
            preprocessed_h5 = h5py.File(preprocessed_path, 'r')
            preprocessed = preprocessed_h5[file_structure.Preprocessed.preprocessed]
            global_parameters[constants.GlobalParameters.input_dimensions] = (preprocessed.shape[1],)
            preprocessed_h5.close()
        else:
            feature_files = list()
            number_features = 0
            number_data_points = 0
            for feature_file_path in global_parameters[constants.GlobalParameters.feature_files]:
                feature_file = h5py.File(feature_file_path, 'r')
                feature_files.append(feature_file)
                number_features += feature_file[file_structure.Preprocessed.preprocessed].shape[1]
                number_data_points = feature_file[file_structure.Preprocessed.preprocessed].shape[0]
            global_parameters[constants.GlobalParameters.input_dimensions] = (number_features,)
            temp_preprocessed_path = file_util.get_temporary_file_path('combined_features')
            preprocessed_h5 = h5py.File(temp_preprocessed_path, 'w')
            preprocessed = hdf5_util.create_dataset(preprocessed_h5, file_structure.Preprocessed.preprocessed,
                                                    (number_data_points, number_features), dtype='float16',
                                                    chunks=(1, number_features))
            offset = 0
            logger.log('Combining features from ' + str(len(feature_files)) + ' sources')
            with progressbar.ProgressBar(len(feature_files)) as progress:
                for feature_file in feature_files:
                    features = feature_file[file_structure.Preprocessed.preprocessed]
                    preprocessed[:, offset:offset + features.shape[1]] = features[:]
                    offset += features.shape[1]
                    feature_file.close()
                    progress.increment()
            preprocessed_h5.close()
            file_util.move_file(temp_preprocessed_path, preprocessed_path)
