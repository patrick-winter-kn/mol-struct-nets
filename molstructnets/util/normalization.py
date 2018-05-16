import h5py
from util import hdf5_util, file_util, logger, progressbar, misc, statistics


class NormalizationTypes:

    min_max_1 = 'Min-Max 0 to 1'
    min_max_2 = 'Min-Max -1 to 1'
    z_score = 'z-Score'


def normalize_data_set(path, data_set_name, type_, stats=None):
    temp_path = file_util.get_temporary_file_path('normalized')
    file_util.copy_file(path, temp_path)
    file_h5 = h5py.File(temp_path, 'r+')
    data_set = file_h5[data_set_name]
    if stats is None:
        chunked_array = misc.get_chunked_array(data_set, fraction=0.5)
        if type_ == NormalizationTypes.min_max_1 or type_ == NormalizationTypes.min_max_2:
            stats = statistics.calculate_statistics(chunked_array, {statistics.Statistics.min,
                                                                    statistics.Statistics.max})
            stats_0 = stats[statistics.Statistics.min]
            stats_1 = stats[statistics.Statistics.max]
        elif type_ == NormalizationTypes.z_score:
            stats = statistics.calculate_statistics(chunked_array, {statistics.Statistics.mean,
                                                                    statistics.Statistics.std})
            stats_0 = stats[statistics.Statistics.mean]
            stats_1 = stats[statistics.Statistics.std]
        chunked_array.close()
        stats = hdf5_util.create_dataset(file_h5, data_set_name + '_normalization_stats', (len(stats_0), 2))
        stats[:, 0] = stats_0[:]
        stats[:, 1] = stats_1[:]
        hdf5_util.set_property(temp_path, data_set_name + '_normalization_type', type_)
    slices = list()
    for length in data_set.shape[:-1]:
        slices.append(slice(0,length))
    logger.log('Normalizing values')
    with progressbar.ProgressBar(stats.shape[0]) as progress:
        for i in range(stats.shape[0]):
            index = tuple(slices + [i])
            if type_ == NormalizationTypes.min_max_1:
                if stats[i, 1] - stats[i, 0] != 0:
                    data_set[index] = (data_set[index] - stats[i, 0]) / (stats[i, 1] - stats[i, 0])
                else:
                    data_set[index] = 0
            elif type_ == NormalizationTypes.min_max_2:
                if stats[i, 1] - stats[i, 0] != 0:
                    data_set[index] = ((2 * (data_set[index] - stats[i, 0])) / (stats[i, 1] - stats[i, 0])) - 1
                else:
                    data_set[index] = 0
            elif type_ == NormalizationTypes.z_score:
                if stats[i, 1] != 0:
                    data_set[index] = (data_set[index] - stats[i, 0]) / stats[i, 1]
                else:
                    data_set[index] = 0
            progress.increment()
    file_h5.close()
    file_util.copy_file(temp_path, path)
