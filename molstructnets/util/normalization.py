import h5py
from util import hdf5_util, file_util, logger, progressbar, misc


def normalize_data_set(path, data_set_name, stats=None):
    temp_path = file_util.get_temporary_file_path('normalized')
    file_util.copy_file(path, temp_path)
    file_h5 = h5py.File(temp_path, 'r+')
    data_set = file_h5[data_set_name]
    if stats is None:
        data_set_read = misc.copy_into_memory(data_set)
        mins = get_mins(data_set_read)
        maxs = get_maxs(data_set_read)
        stats = hdf5_util.create_dataset(file_h5, data_set_name + '_normalization_stats', (len(mins), 2))
        stats[:, 0] = mins[:]
        stats[:, 1] = maxs[:]
    slices = list()
    for length in data_set.shape[:-1]:
        slices.append(slice(0,length))
    logger.log('Normalizing values')
    with progressbar.ProgressBar(stats.shape[0]) as progress:
        for i in range(stats.shape[0]):
            index = tuple(slices + [i])
            if stats[i, 0] - stats[i, 1] != 0:
                data_set[index] = ((2 * (data_set[index] - stats[i, 0])) / (stats[i, 1] - stats[i, 0])) - 1
            else:
                data_set[index] = 0
            progress.increment()
    file_h5.close()
    file_util.copy_file(temp_path, path)


def get_mins(data_set):
    return data_set.min(tuple(range(len(data_set.shape) - 1)))


def get_maxs(data_set):
    return data_set.max(tuple(range(len(data_set.shape) - 1)))
