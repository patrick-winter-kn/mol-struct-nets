import argparse

from util import initialization


def get_arguments():
    parser = argparse.ArgumentParser(description='Creates a 360° CAM for the given SMILES string')
    parser.add_argument('network', type=str, help='Path to the network file')
    parser.add_argument('preprocessing', type=str, help='Path to the preprocessing file')
    parser.add_argument('smiles', type=str, help='The SMILES string')
    parser.add_argument('output', type=str, help='The output folder')
    return parser.parse_args()


args = get_arguments()
initialization.initialize(args)

import gc
import time
from util import file_util, progressbar, file_structure
from steps.preprocessing.shared.tensor2d import tensor_2d_preprocessor
import numpy
from steps.interpretation.shared.kerasviz import cam
from keras import models
from steps.interpretation.shared import tensor_2d_renderer
import h5py

smiles = args.smiles
network_path = file_util.resolve_path(args.network)
network = models.load_model(network_path)
preprocessing_path = file_util.resolve_path(args.preprocessing)
preprocessed_h5 = h5py.File(preprocessing_path, 'r')
symbols = preprocessed_h5[file_structure.PreprocessedTensor2D.symbols][:]
preprocessed_h5.close()
preprocessor = tensor_2d_preprocessor.Tensor2DPreprocessor(preprocessing_path)
output_dir_path = file_util.resolve_subpath(args.output, smiles)
file_util.make_folders(output_dir_path, including_this=True)
cam_calc = cam.CAM(network_path, 0)
with progressbar.ProgressBar(360) as progress:
    for i in range(360):
        preprocessed = preprocessor.preprocess_single_smiles(smiles, rotation=i)
        tensor = numpy.zeros([1] + list(preprocessor.shape), dtype='float32')
        preprocessed.fill_array(tensor)
        prediction = network.predict(tensor)[0][0]
        grads = cam_calc.calculate(tensor[0])
        cam_data = grads[:] * prediction
        heatmap = cam.array_to_heatmap([cam_data])[0]
        output_path = file_util.resolve_subpath(output_dir_path, str(i) + '.svgz')
        tensor_2d_renderer.render(output_path, tensor[0], symbols, heatmap=heatmap)
        progress.increment()

gc.collect()
time.sleep(1)
