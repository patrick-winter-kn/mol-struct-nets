import os
import sys
import random
import numpy
import json
import matplotlib


silent_loading = True


def initialize(args):
    global seed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['PYTHONHASHSEED'] = '0'
    seed = random.randint(0, numpy.iinfo(numpy.uint32).max)
    if args.seed is not None:
        seed = args.seed
    else:
        experiment_path = os.path.abspath(args.experiment)
        if os.path.exists(experiment_path):
            dict_ = json.load(open(experiment_path))
            if 'seed' in dict_:
                seed = dict_['seed']
    random.seed(seed)
    numpy.random.seed(seed)
    matplotlib.use('Agg')

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
