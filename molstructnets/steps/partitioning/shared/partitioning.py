from util import logger, progressbar, hdf5_util
import math


def oversample(partition_h5, data_set_name, classes):
    logger.log('Oversampling data')
    ref = partition_h5[data_set_name]
    class_zero_count = 0
    class_one_count = 0
    for i in range(ref.shape[0]):
        value = classes[ref[i]]
        if value[0] >= value[1]:
            class_zero_count += 1
        else:
            class_one_count += 1
    difference = abs(class_zero_count - class_one_count)
    oversampled = hdf5_util.create_dataset(partition_h5, data_set_name + '-oversampled', (ref.shape[0] + difference,),
                                           dtype='I')
    left_difference = difference
    if class_zero_count < class_one_count:
        copies_per_instance = int(math.ceil(class_one_count / class_zero_count))
    else:
        copies_per_instance = int(math.ceil(class_zero_count / class_one_count))
    target_i = 0
    with progressbar.ProgressBar(oversampled.shape[0]) as progress:
        for i in range(len(ref)):
            value = classes[ref[i]]
            minority = (class_zero_count < class_one_count and value[0] >= value[1]) or \
                       (class_one_count < class_zero_count and value[1] > value[0])
            copies = 1
            if left_difference > 0 and minority:
                copies = min(left_difference + 1, copies_per_instance)
                left_difference -= copies - 1
            for j in range(copies):
                oversampled[target_i] = ref[i]
                target_i += 1
                progress.increment()
    del partition_h5[data_set_name]
    partition_h5[data_set_name] = partition_h5[data_set_name + '-oversampled']
    del partition_h5[data_set_name + '-oversampled']
    return partition_h5[data_set_name]


def shuffle(data_set, random_):
    n = len(data_set)
    logger.log('Shuffling data')
    with progressbar.ProgressBar(n) as progress:
        for i in range(n):
            j = random_.randint(0, n - 1)
            tmp = data_set[j]
            data_set[j] = data_set[i]
            data_set[i] = tmp
            progress.increment()
