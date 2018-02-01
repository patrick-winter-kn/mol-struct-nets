import numpy
import math
from matplotlib import pyplot
from util import logger, progressbar, file_util, misc
import random


def plot(predictions_list, prediction_names, classes, roc_curve_plot_file, shuffle=True, seed=42):
    actives_list = []
    inactives_list = []
    auc_list = []
    # We copy the needed data into memory to speed up sorting
    classes = misc.copy_into_memory(classes, as_bool=True)
    for i in range(len(predictions_list)):
        logger.log('Calculating stats for ' + prediction_names[i], logger.LogLevel.VERBOSE)
        actives, inactives, auc= stats(predictions_list[i], classes, shuffle=shuffle, seed=seed)
        actives_list.append(actives)
        inactives_list.append(inactives)
        auc_list.append(auc)
    axis = pyplot.subplots()[1]
    axis.grid(True, linestyle='--')
    # Plot random line
    pyplot.plot((0, 1), (0, actives[-1]), ls='-', c='0.75')
    # Plot actives
    for i in range(len(predictions_list)):
        pyplot.plot(inactives_list[i], actives_list[i],
                    label=prediction_names[i]+' (AUC: ' + str(round(auc_list[i], 2)) + ')')
    pyplot.ylabel('True Positive Rate')
    pyplot.xlabel('False Positive Rate')
    pyplot.legend(loc='lower right', fancybox=True)
    pyplot.tight_layout()
    if roc_curve_plot_file:
        file_util.make_folders(roc_curve_plot_file)
        pyplot.savefig(roc_curve_plot_file, format='svgz', transparent=True)
    else:
        pyplot.show()
    pyplot.close('all')
    return auc_list


def stats(predictions, classes, positives=None, shuffle=True, seed=42):
    if positives is None:
        positives = positives_count(classes)
    negatives = len(classes) - positives
    # We copy the needed data into memory to speed up sorting
    classes = misc.copy_into_memory(classes, as_bool=True)
    predictions = misc.copy_into_memory(predictions)
    # First axis of first element
    predictions = predictions[:, 0]
    if shuffle:
        shuffle_indices = list(range(len(classes)))
        random.Random(seed).shuffle(shuffle_indices)
        classes = classes[shuffle_indices, ...]
        predictions = predictions[shuffle_indices, ...]
    # Sort it (.argsort()) and reverse the order ([::-1]))
    indices = predictions.argsort()[::-1]
    actives = [0]
    inactives = [0]
    found_active = 0
    found_inactive = 0
    curve_sum = 0
    logger.log('Calculating ROC curve stats', logger.LogLevel.VERBOSE)
    with progressbar.ProgressBar(len(indices), logger.LogLevel.VERBOSE) as progress:
        for i in range(len(indices)):
            row = classes[indices[i]]
            # Check if index (numpy.where) of maximum value (max(row)) in row is 0 (==0)
            # This means the active value is higher than the inactive value
            if numpy.where(row == max(row))[0] == 0:
                found_active += 1
            else:
                found_inactive += 1
                curve_sum += found_active
            actives.append(found_active / positives)
            inactives.append(found_inactive / negatives)
            progress.increment()
    auc = (curve_sum / positives) / negatives
    logger.log('AUC: ' + str(auc), logger.LogLevel.VERBOSE)
    return actives, inactives, auc


def positives_count(classes):
    positives = 0
    logger.log('Counting actives')
    with progressbar.ProgressBar(len(classes)) as progress:
        i = 0
        for row in classes:
            if numpy.where(row == max(row))[0] == 0:
                positives += 1
            i += 1
            progress.increment()
    logger.log('Found ' + str(positives) + ' actives and ' + str(len(classes) - positives) + ' inactives',
               logger.LogLevel.VERBOSE)
    return positives
