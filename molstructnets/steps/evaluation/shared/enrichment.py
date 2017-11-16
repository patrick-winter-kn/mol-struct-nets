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
    classes = misc.copy_into_memory(classes, as_bool=True)
    for i in range(len(predictions_list)):
        logger.log('Calculating stats for ' + prediction_names[i], logger.LogLevel.VERBOSE)
        actives, auc, efs = stats(predictions_list[i], classes, enrichment_factors, shuffle=shuffle, seed=seed)
        actives_list.append(actives)
        efs_list.append(efs)
        auc_list.append(auc)
    axis = pyplot.subplots()[1]
    axis.grid(True, linestyle='--')
    # Plot ideal line
    pyplot.plot((0, actives[-1]), (0, actives[-1]), ls='-', c='0.75')
    pyplot.plot((actives[-1], len(actives) - 1), (actives[-1], actives[-1]), ls='-', c='0.75')
    # Plot random line
    pyplot.plot((0, len(actives) - 1), (0, actives[-1]), ls='-', c='0.75')
    # Plot actives
    for i in range(len(predictions_list)):
        pyplot.plot(actives_list[i], label=prediction_names[i]+' (AUC: ' + str(round(auc_list[i], 2)) + ')')
    # Add enrichment factors
    for percent in sorted(enrichment_factors):
        x = percent * 0.01 * len(classes)
        y_end = 0
        for actives in actives_list:
            y_end = max(y_end, calculate_y_at_x(x, actives))
        ef_label = 'Enrichment factor ' + str(percent) + '%'
        for i in range(len(efs_list)):
            ef_label += '\n' + prediction_names[i] + ': ' + str(round(efs_list[i][percent], 2))
        pyplot.plot((x, x), (0, y_end), ls='--', label=ef_label)
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
    found = 0
    curve_sum = 0
    logger.log('Calculating enrichment stats', logger.LogLevel.VERBOSE)
    with progressbar.ProgressBar(len(indices), logger.LogLevel.VERBOSE) as progress:
        for i in range(len(indices)):
            row = classes[indices[i]]
            curve_sum += found
            # Check if index (numpy.where) of maximum value (max(row)) in row is 0 (==0)
            # This means the active value is higher than the inactive value
            if numpy.where(row == max(row))[0] == 0:
                curve_sum += 0.5
                found += 1
            actives.append(found)
            progress.increment()
    # AUC = sum of found positives for every x / (positives * (number of samples + 1))
    # + 1 is added to the number of samples for the start with 0 samples selected
    auc = curve_sum / (positives * len(classes))
    logger.log('AUC: ' + str(auc), logger.LogLevel.VERBOSE)
    # Calculate enrichment factor by dividing the number of found positives by the number of positives found at random
    efs = {}
    for percent in ef_percent:
        # Calculate x position
        x = percent * 0.01 * len(classes)
        efs[percent] = calculate_y_at_x(x, actives) / (percent * 0.01 * actives[-1])
        logger.log('EF at ' + str(percent) + '%: ' + str(efs[percent]), logger.LogLevel.VERBOSE)
    return actives, auc, efs


def calculate_y_at_x(x, actives, at_random=False):
    if at_random:
        return x * actives[-1]
    else:
        # Integer x before / at x
        previous_x = math.floor(x)
        # Integer x after x
        next_x = previous_x + 1
        # How far is x on the way to next_x
        between = x % 1
        # Difference of y of previous_x and next_x
        difference = actives[next_x] - actives[previous_x]
        # Increase of y for x between previous and next x
        increase = between * difference
        # y at position x for predictions
        return actives[previous_x] + increase


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
