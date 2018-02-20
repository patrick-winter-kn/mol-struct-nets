from util import initialization
import argparse
from steps.preprocessing.matrix2d import matrix_2d
from steps.preprocessing.shared.matrix2d import molecule_2d_matrix, rasterizer
from rdkit import Chem
from steps.interpretation.shared import matrix_2d_renderer
from steps.preprocessingtraining.matrix2dtransformation import transformer


def get_arguments():
    parser = argparse.ArgumentParser(description='Renders the given SMILES in 2D')
    parser.add_argument('smiles', type=str, help='The SMILES string')
    parser.add_argument('path', type=str, help='The path to the output file')
    parser.add_argument('--scale', type=float, default=2.0, help='Scaling factor')
    parser.add_argument('--square', type=bool, default=True, help='If high should equal width')
    parser.add_argument('--padding', type=int, default=molecule_2d_matrix.padding, help='Padding around the edges')
    parser.add_argument('--rotation', type=int, default=0, help='Angle for rotation')
    parser.add_argument('--flip', type=bool, default=False, help='If the layout should be flipped')
    return parser.parse_args()

args = get_arguments()
symbols = set()
max_nr_atoms = set()
min_x = set()
min_y = set()
max_x = set()
max_y = set()
matrix_2d.Matrix2D.analyze_smiles([args.smiles.encode('utf-8')], symbols, max_nr_atoms, min_x, min_y, max_x, max_y,
                                  None)
molecule = Chem.MolFromSmiles(args.smiles, sanitize=False)
symbols = sorted(symbols | matrix_2d.fixed_symbols)
index_lookup = {}
symbol_index = list()
for i in range(len(symbols)):
    index_lookup[symbols[i]] = i
    symbol_index.append(symbols[i].encode('utf-8'))
min_x = min(min_x)
min_y = min(min_y)
max_x = max(max_x)
max_y = max(max_y)
rasterizer_ = rasterizer.Rasterizer(args.scale, args.padding, min_x, max_x, min_y, max_y, args.square)
transformer_ = transformer.Transformer(min_x, max_x, min_y, max_y)
preprocessed_shape = (1, rasterizer_.size_x, rasterizer_.size_y, len(index_lookup))
preprocessed_row =\
    molecule_2d_matrix.molecule_to_2d_matrix(molecule, index_lookup, rasterizer_, preprocessed_shape,
                                             atom_locations_shape=None, transformer_=transformer_, random_=None,
                                             flip=args.flip, rotation=args.rotation)[0]
matrix_2d_renderer.render(args.path, preprocessed_row, symbol_index, render_factor=50, show_grid=True, heatmap=None,
                          background_heatmap=True)
