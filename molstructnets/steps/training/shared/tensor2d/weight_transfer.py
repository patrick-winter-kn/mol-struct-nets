from util import misc


def get_weight_range(model, start_layer_name, end_layer_name):
    start_index = None
    end_index = None
    names = None
    for i in range(len(model.layers)):
        name = model.layers[i].name
        if name == start_layer_name:
            names = set()
        if names is not None:
            names.add(name)
        if name == end_layer_name:
            break
    for i in range(len(model.weights)):
        name = model.weights[i].name
        name = name[:name.rfind('_')]
        if name in names:
            start_index = misc.minimum(start_index, i)
            end_index = misc.maximum(end_index, i)
    return start_index, end_index + 1


def transfer_weights(source_model, destination_model, start_index, end_index):
    destination_weights = destination_model.get_weights()
    destination_weights[start_index:end_index] = source_model.get_weights()[start_index:end_index]
    destination_model.set_weights(destination_weights)


def set_weight_freeze(model, start_index, end_index, freeze):
    for i in range(start_index, end_index):
        model.layers[i].trainable = not freeze
    model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
