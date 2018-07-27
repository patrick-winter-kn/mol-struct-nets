# The methods in this class are modified versions of methods in the keras-vis plugin. See LICENSE.txt for license
# information.

import numpy
from keras import models, activations
from matplotlib import cm
from vis.utils import utils
from vis.visualization import saliency


class CAM():
    # This class does the same as viz.visualization.saliency.visualize_saliency() but it returns the original values
    # instead of the heatmap

    def __init__(self, model_path, class_index):
        model = models.load_model(model_path)
        out_layer_index = len(model.layers) - 1
        model.layers[out_layer_index].activation = activations.linear
        model = utils.apply_modifications(model)
        losses = [(saliency.ActivationMaximization(model.layers[out_layer_index], [class_index]), -1)]
        self._opt = saliency.Optimizer(model.input, losses, norm_grads=False)

    def calculate(self, input_data):
        grads = self._opt.minimize(seed_input=input_data, max_iter=1, grad_modifier='absolute', verbose=False)[1]
        grads = numpy.max(grads, axis=-1)
        grads = utils.normalize(grads)
        return grads


def array_to_heatmap(array):
    return numpy.uint8(cm.jet(array)[..., :3] * 255)[0]
