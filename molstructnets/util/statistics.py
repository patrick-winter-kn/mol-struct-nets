import numpy


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


def calculate_statistics(array, stats):
    stats = set(stats)
    if Statistics.std in stats:
        stats.add(Statistics.mean)
    statistics = dict()
    if isinstance(array, numpy.ndarray):
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
        for stat in stats:
            if stat == Statistics.min:
                statistics[stat] = numpy.ndarray((array.number_chunks(), array.shape[-1]))
            if stat == Statistics.max:
                statistics[stat] = numpy.ndarray((array.number_chunks(), array.shape[-1]))
            if stat == Statistics.mean:
                statistics[stat] = numpy.ndarray((array.number_chunks(), array.shape[-1]))
            if stat == Statistics.std:
                statistics[stat] = numpy.ndarray((array.number_chunks(), array.shape[-1]))
        more = True
        while more:
            current_chunk_number = array.get_current_chunk_number()
            for stat in stats:
                if stat == Statistics.min:
                    statistics[stat][current_chunk_number,:] = array.min(tuple(range(len(array.shape) - 1)))
                if stat == Statistics.max:
                    statistics[stat][current_chunk_number,:] = array.max(tuple(range(len(array.shape) - 1)))
                if stat == Statistics.mean:
                    statistics[stat][current_chunk_number,:] = array.mean(tuple(range(len(array.shape) - 1)))
            more = array.has_next()
            if more:
                array.load_next_chunk()
        for stat in stats:
            if stat == Statistics.min:
                statistics[stat] = statistics[stat].min(0)
            if stat == Statistics.max:
                statistics[stat] = statistics[stat].max(0)
            if stat == Statistics.mean:
                for i in range(statistics[stat].shape[0]):
                    statistics[stat][i] = statistics[stat][i] * array.get_chunks()[i]['size']
                statistics[stat] = statistics[stat].sum(0) / array.get_overall_size()
        if Statistics.std in stats:
            array.reset()
            more = True
            sums = numpy.ndarray(array.shape[-1])
            while more:
                slices = list()
                for length in array.shape[:-1]:
                    slices.append(slice(0,length))
                for i in range(array.shape[-1]):
                    index = tuple(slices + [i])
                    a = array[index]
                    a -= statistics[Statistics.mean][i]
                    a **= 2
                    sums[i] += a.sum()
                more = array.has_next()
                if more:
                    array.load_next_chunk()
            sums /= array.get_overall_size()
            statistics[Statistics.std] = numpy.sqrt(sums)
    return statistics
