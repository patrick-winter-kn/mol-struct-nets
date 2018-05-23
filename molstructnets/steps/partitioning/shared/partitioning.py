from util import logger, progressbar, hdf5_util
import math
import numpy


def oversample(partition, classes, log_level=logger.LogLevel.INFO):
    logger.log('Oversampling data', log_level=log_level)
    class_zero_count = int(classes[partition,0].sum(0))
    class_one_count = int(classes[partition,1].sum(0))
    if class_zero_count == 0 or class_one_count == 0:
        raise ValueError('One of the classes is not represented')
    difference = abs(class_zero_count - class_one_count)
    oversampled_partition = numpy.zeros(partition.shape[0] + difference, dtype='uint32')
    left_difference = difference
    if class_zero_count < class_one_count:
        copies_per_instance = int(math.ceil(class_one_count / class_zero_count))
    else:
        copies_per_instance = int(math.ceil(class_zero_count / class_one_count))
    target_i = 0
    with progressbar.ProgressBar(oversampled_partition.shape[0], log_level) as progress:
        for i in range(len(partition)):
            value = classes[partition[i]]
            minority = (class_zero_count < class_one_count and value[0] >= value[1]) or \
                       (class_one_count < class_zero_count and value[1] > value[0])
            copies = 1
            if left_difference > 0 and minority:
                copies = min(left_difference + 1, copies_per_instance)
                left_difference -= copies - 1
            for j in range(copies):
                oversampled_partition[target_i] = partition[i]
                target_i += 1
                progress.increment()
    return oversampled_partition


def shuffle(data, random_, log_level=logger.LogLevel.INFO):
    n = len(data)
    logger.log('Shuffling data', log_level)
    with progressbar.ProgressBar(n, log_level) as progress:
        for i in range(n):
            j = random_.randint(0, n - 1)
            tmp = data[j]
            data[j] = data[i]
            data[i] = tmp
            progress.increment()
