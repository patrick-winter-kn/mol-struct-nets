import os
import pathlib

cuda_devices_file = str(pathlib.Path.home()) + os.sep + '.cuda_devices'
if os.path.isfile(cuda_devices_file):
    with open(cuda_devices_file, 'r') as value_file:
        os.environ['CUDA_VISIBLE_DEVICES'] = value_file.read().replace('\n', '')

import sys
import random
import numpy
import json
import matplotlib

silent_loading = True
seed = 1


def initialize(args=None):
    global seed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['PYTHONHASHSEED'] = '0'
    seed = random.randint(0, numpy.iinfo(numpy.uint32).max)
    if hasattr(args, 'seed') and args.seed is not None:
        seed = args.seed
    elif hasattr(args, 'experiment') and args.experiment is not None:
        experiment_path = os.path.abspath(args.experiment)
        if os.path.exists(experiment_path):
            dict_ = json.load(open(experiment_path))
            if 'seed' in dict_:
                seed = dict_['seed']
    random.seed(seed)
    numpy.random.seed(seed)
    matplotlib.use('Agg')

    stdout = None
    stderr = None
    if silent_loading:
        stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')

    import tensorflow
    config = tensorflow.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.75
    session = tensorflow.Session(config=config)
    from keras.backend import tensorflow_backend
    tensorflow_backend.set_session(session)

    if silent_loading:
        sys.stdout = stdout
        sys.stderr = stderr
