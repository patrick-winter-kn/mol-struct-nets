import numpy
from util import logger, progressbar


class Statistics:

    min = 'minimum'
    max = 'maximum'
    mean = 'mean'
    std = 'std_deviation'


def get_mins(data_set):
    return data_set.min(tuple(range(len(data_set.shape) - 1)))


def get_maxs(data_set):
    return data_set.max(tuple(range(len(data_set.shape) - 1)))


def get_means(data_set):
    return data_set.mean(tuple(range(len(data_set.shape) - 1)))


def get_stds(data_set):
    return data_set.std(tuple(range(len(data_set.shape) - 1)))


def calculate_additional_memory(shape, stats):
    if Statistics.std in stats:
        return 1/shape[-1]
    else:
        return 0


def calculate_statistics(array, stats, log_level=logger.LogLevel.INFO):
    stats = set(stats)
    if Statistics.std in stats:
        stats.add(Statistics.mean)
    statistics = dict()
    if isinstance(array, numpy.ndarray):
        logger.log('Calculating statistics: ' + str(sorted(stats)), log_level)
        for stat in stats:
            if stat == Statistics.min:
                statistics[stat] = get_mins(array)
            if stat == Statistics.max:
                statistics[stat] = get_maxs(array)
            if stat == Statistics.mean:
                statistics[stat] = get_means(array)
            if stat == Statistics.std:
                statistics[stat] = get_stds(array)
    else:
        logger.log('Calculating statistics: ' + str(sorted(stats))
                   + ' (in ' + str(len(array.get_chunks())) + ' chunks)', log_level)
        number_first_run = len(stats)
        number_second_run = 0
        if Statistics.std in stats:
            number_first_run -= 1
            number_second_run += 1
        number_first_run = number_first_run * array.number_chunks() + array.number_chunks()
        if number_second_run > 0:
                number_second_run = number_second_run * array.number_chunks() + array.number_chunks()
        with progressbar.ProgressBar(number_first_run + number_second_run, log_level) as progress:
            for stat in stats:
                if stat == Statistics.min:
                    statistics[stat] = numpy.ndarray((array.number_chunks(), array.original_shape[-1]))
                if stat == Statistics.max:
                    statistics[stat] = numpy.ndarray((array.number_chunks(), array.original_shape[-1]))
                if stat == Statistics.mean:
                    statistics[stat] = numpy.ndarray((array.number_chunks(), array.original_shape[-1]))
                if stat == Statistics.std:
                    statistics[stat] = numpy.ndarray((array.number_chunks(), array.original_shape[-1]))
            # First run
            for i in range(array.number_chunks()):
                array.load_chunk(i)
                progress.increment()
                current_chunk_number = array.get_current_chunk_number()
                for stat in stats:
                    if stat == Statistics.min:
                        statistics[stat][current_chunk_number, :] = array[:].min(tuple(range(len(array.shape) - 1)))
                        progress.increment()
                    if stat == Statistics.max:
                        statistics[stat][current_chunk_number, :] = array[:].max(tuple(range(len(array.shape) - 1)))
                        progress.increment()
                    if stat == Statistics.mean:
                        statistics[stat][current_chunk_number, :] = array[:].mean(tuple(range(len(array.shape) - 1)))
                        progress.increment()
            for stat in stats:
                if stat == Statistics.min:
                    statistics[stat] = statistics[stat].min(0)
                if stat == Statistics.max:
                    statistics[stat] = statistics[stat].max(0)
                if stat == Statistics.mean:
                    for i in range(statistics[stat].shape[0]):
                        statistics[stat][i] = statistics[stat][i] * array.get_chunks()[i]['size']
                    statistics[stat] = statistics[stat].sum(0) / array.original_shape[0]
            if Statistics.std in stats:
                sums = numpy.ndarray(array.original_shape[-1])
                # Second run
                for i in reversed(range(array.number_chunks())):
                    progress.increment()
                    slices = list()
                    for length in array.shape[:-1]:
                        slices.append(slice(0,length))
                    for j in range(array.shape[-1]):
                        index = tuple(slices + [j])
                        a = array[index].copy()
                        a -= statistics[Statistics.mean][j]
                        a **= 2
                        sums[j] += a.sum()
                        a = None
                    progress.increment()
                sums /= array.original_shape[0]
                statistics[Statistics.std] = numpy.sqrt(sums)
    return statistics
