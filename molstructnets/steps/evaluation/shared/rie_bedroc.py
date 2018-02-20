import numpy
from rdkit.ML.Scoring import Scoring
from util import logger, progressbar, file_util, misc
import random


def stats(predictions, classes, alphas, positives=None, shuffle=True, seed=42):
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
    indices = predictions.argsort()[::-1]
    ries = list()
    bedrocs = list()
    for alpha in alphas:
        ries.append(Scoring.CalcRIE(classes[indices], 0, alpha))
        bedrocs.append(Scoring.CalcBEDROC(classes[indices], 0, alpha))
    return ries, bedrocs