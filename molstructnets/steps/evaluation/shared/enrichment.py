import numpy
import math
from matplotlib import pyplot
from util import logger, progressbar, file_util, misc
import random


def plot(predictions_list, prediction_names, classes, enrichment_factors, enrichment_plot_file, shuffle=True, seed=42):
    actives_list = []
    efs_list = []
    auc_list = []
    # We copy the needed data into memory to speed up sorting
    classes = misc.copy_ndarray(classes)
    for i in range(len(predictions_list)):
        logger.log('Calculating stats for ' + prediction_names[i], logger.LogLevel.VERBOSE)
        actives, auc, efs = stats(predictions_list[i], classes, enrichment_factors, shuffle, seed)
        actives_list.append(actives)
        efs_list.append(efs)
        auc_list.append(auc)
    axis = pyplot.subplots()[1]
    axis.grid(True, linestyle='--')
    # Plot ideal line
    pyplot.plot((0, actives[-1]), (0, actives[-1]), ls='-', c='0.75')
    pyplot.plot((actives[-1], len(actives)), (actives[-1], actives[-1]), ls='-', c='0.75')
    # Plot random line
    pyplot.plot((0,len(actives)), (0,actives[-1]), ls='-', c='0.75')
    # Plot actives
    for i in range(len(predictions_list)):
        pyplot.plot(actives_list[i], label=prediction_names[i]+' (AUC: ' + str(round(auc_list[i], 2)) + ')')
    # Add enrichment factors
    for percent in sorted(enrichment_factors):
        x_start_end = int(math.ceil(percent * 0.01 * len(classes)))
        y_end = 0
        for actives in actives_list:
            y_end = max(y_end, actives[x_start_end])
        ef_label = 'Enrichment factor ' + str(percent) + '%'
        for i in range(len(efs_list)):
            ef_label += '\n' + prediction_names[i] + ': ' + str(round(efs_list[i][percent], 2))
        pyplot.plot((x_start_end, x_start_end), (0, y_end), ls='--', label=ef_label)
    pyplot.ylabel('Active Compounds')
    pyplot.xlabel('Compounds')
    pyplot.legend(loc='lower right', fancybox=True)
    pyplot.tight_layout()
    if enrichment_plot_file:
        file_util.make_folders(enrichment_plot_file)
        pyplot.savefig(enrichment_plot_file, format='svg', transparent=True)
    else:
        pyplot.show()
    pyplot.close('all')
    return auc_list, efs_list


def stats(predictions, classes, ef_percent, positives=None, shuffle=True, seed=42):
    if positives is None:
        positives = positives_count(classes)
    # We copy the needed data into memory to speed up sorting
    if not isinstance(classes, numpy.ndarray):
        classes = misc.copy_ndarray(classes)
    if not isinstance(predictions, numpy.ndarray):
        predictions = misc.copy_ndarray(predictions)
    # First axis of first element
    predictions = predictions[:,0]
    if shuffle:
        shuffle_indices = list(range(len(classes)))
        random.Random(seed).shuffle(shuffle_indices)
        classes = classes[shuffle_indices, ...]
        predictions = predictions[shuffle_indices, ...]
    # Sort it (.argsort()) and reverse the order ([::-1]))
    indices = predictions.argsort()[::-1]
    actives = [0]
    # efs maps the percent to the number of found positives
    efs = {}
    for percent in ef_percent:
        efs[percent] = 0
    found = 0
    curve_sum = 0
    logger.log('Calculating enrichment stats', logger.LogLevel.VERBOSE)
    with progressbar.ProgressBar(len(indices), logger.LogLevel.VERBOSE) as progress:
        for i in range(len(indices)):
            row = classes[indices[i]]
            # Check if index (numpy.where) of maximum value (max(row)) in row is 0 (==0)
            # This means the active value is higher than the inactive value
            if numpy.where(row == max(row))[0] == 0:
                found += 1
                for percent in efs.keys():
                    # If i is still part of the fraction count the number of founds up
                    if i < int(math.floor(len(indices)*(percent*0.01))):
                        efs[percent] += 1
            curve_sum += found
            actives.append(found)
            progress.increment()
    # AUC = sum of found positives for every x / (positives * (number of samples + 1))
    # + 1 is added to the number of samples for the start with 0 samples selected
    auc = curve_sum / (positives * (len(classes) + 1))
    logger.log('AUC: ' + str(auc), logger.LogLevel.VERBOSE)
    # Turn number of found positives into enrichment factor by dividing the number of positives found at random
    for percent in sorted(efs.keys()):
        efs[percent] /= (positives * (percent * 0.01))
        logger.log('EF at ' + str(percent) + '%: ' + str(efs[percent]), logger.LogLevel.VERBOSE)
    return actives, auc, efs


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
    logger.log('Found ' + str(positives) + ' actives and ' + str(len(classes) - positives) + ' inactives', logger.LogLevel.VERBOSE)
    return positives
