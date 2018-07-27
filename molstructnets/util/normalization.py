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
        if type_ == NormalizationTypes.min_max_1 or type_ == NormalizationTypes.min_max_2:
            stats_set = {statistics.Statistics.min, statistics.Statistics.max}
            additional_percent = statistics.calculate_additional_memory(data_set.shape, stats_set)
            fraction = 1 / (1 + additional_percent)
            chunked_array = misc.get_chunked_array(data_set, fraction=fraction)
            stats = statistics.calculate_statistics(chunked_array, stats_set)
            stats_0 = stats[statistics.Statistics.min]
            stats_1 = stats[statistics.Statistics.max]
        elif type_ == NormalizationTypes.z_score:
            stats_set = {statistics.Statistics.mean, statistics.Statistics.std}
            additional_percent = statistics.calculate_additional_memory(data_set.shape, stats_set)
            fraction = 1 / (1 + additional_percent)
            chunked_array = misc.get_chunked_array(data_set, fraction=fraction)
            stats = statistics.calculate_statistics(chunked_array, stats_set)
            stats_0 = stats[statistics.Statistics.mean]
            stats_1 = stats[statistics.Statistics.std]
        stats = hdf5_util.create_dataset(file_h5, data_set_name + '_normalization_stats', (len(stats_0), 2))
        stats[:, 0] = stats_0[:]
        stats[:, 1] = stats_1[:]
        hdf5_util.set_property(temp_path, data_set_name + '_normalization_type', type_)
    else:
        chunked_array = misc.get_chunked_array(data_set, fraction=1)
    logger.log('Normalizing values of ' + str(stats.shape[0]) + ' features (in ' + str(len(chunked_array.get_chunks()))
               + ' chunks)')
    with progressbar.ProgressBar(3 * chunked_array.number_chunks()) as progress:
        for i in range(chunked_array.number_chunks()):
            chunked_array.load_chunk(i)
            progress.increment()
            normalized = chunked_array[:]
            slices = list()
            for length in normalized.shape[:-1]:
                slices.append(slice(0, length))
            for j in range(stats.shape[0]):
                index = tuple(slices + [j])
                if type_ == NormalizationTypes.min_max_1:
                    if stats[j, 1] - stats[j, 0] != 0:
                        normalized[index] -= stats[j, 0]
                        normalized[index] /= stats[j, 1] - stats[j, 0]
                    else:
                        normalized[index] = 0
                elif type_ == NormalizationTypes.min_max_2:
                    if stats[j, 1] - stats[j, 0] != 0:
                        normalized[index] -= stats[j, 0]
                        normalized[index] *= 2
                        normalized[index] /= stats[j, 1] - stats[j, 0]
                        normalized[index] -= 1
                    else:
                        normalized[index] = 0
                elif type_ == NormalizationTypes.z_score:
                    if stats[j, 1] != 0:
                        normalized[index] -= stats[j, 0]
                        normalized[index] /= stats[j, 1]
                    else:
                        normalized[index] = 0
            progress.increment()
            chunked_array.write_current_chunk()
            progress.increment()
            normalized = None
    file_h5.close()
    file_util.copy_file(temp_path, path)
