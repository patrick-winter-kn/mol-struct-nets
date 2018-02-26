# The methods in this class are modified versions of methods in the keras-vis plugin. See LICENSE.txt for license
# information.

from vis.visualization import saliency
from keras import backend
import numpy
from vis.utils import utils
from matplotlib import cm


def calculate_saliency(model, layer_idx, filter_indices, seed_input):
    # This method does the same as viz.visualization.saliency.visualize_saliency() but it returns the original values
    # instead of the heatmap
    losses = [(saliency.ActivationMaximization(model.layers[layer_idx], filter_indices), -1)]
    opt = saliency.Optimizer(model.input, losses, norm_grads=False)
    grads = opt.minimize(seed_input=seed_input, max_iter=1, grad_modifier='absolute', verbose=False)[1]
    channel_idx = 1 if backend.image_data_format() == 'channels_first' else -1
    grads = numpy.max(grads, axis=channel_idx)
    grads = utils.normalize(grads)
    return grads


def array_to_heatmap(array):
    return numpy.uint8(cm.jet(array)[..., :3] * 255)[0]
