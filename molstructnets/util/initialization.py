import os
import sys
import random
import numpy
import json
import matplotlib


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '0'
seed = random.randint(0, numpy.iinfo(numpy.uint32).max)
experiment_path = os.path.abspath(sys.argv[1])
if os.path.exists(experiment_path):
    dict_ = json.load(open(experiment_path))
    if 'seed' in dict_:
        seed = dict_['seed']
random.seed(seed)
numpy.random.seed(seed)
matplotlib.use('Agg')
