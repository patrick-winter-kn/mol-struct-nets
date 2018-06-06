from sklearn import ensemble
from sklearn.externals import joblib
import numpy
from util import file_util, misc
import math


def train(train_data_input, train_data_output, model_path, nr_trees=1000, min_samples_leaf=1, seed=None):
    if len(train_data_input.shape) > 2:
        nr_features = numpy.prod(train_data_input.shape[1:])
        train_data_input = train_data_input.reshape((train_data_input.shape[0], nr_features))
    train_data_output = train_data_output[:,1]
    random_forest = ensemble.RandomForestClassifier(n_estimators=nr_trees, min_samples_leaf=min_samples_leaf, n_jobs=-1,
                                                    class_weight='balanced', verbose=1, criterion='gini',
                                                    random_state=seed)
    random_forest.fit(train_data_input, train_data_output)
    file_util.make_folders(model_path)
    joblib.dump(random_forest, model_path)


def predict(test_data_input, model):
    if len(test_data_input.shape) > 2:
        nr_features = numpy.prod(test_data_input.shape[1:])
        test_data_input = test_data_input.reshape((test_data_input.shape[0], nr_features))
    if isinstance(model, str):
        model = joblib.load(model)
    probabilities = model.predict_proba(test_data_input)
    probabilities = numpy.array(probabilities)
    return probabilities
