import re

import h5py
from rdkit import Chem

from steps.interpretation.shared import smiles_renderer
from util import data_validation, file_structure, file_util, progressbar, hdf5_util


class SubstructureHighlight:

    rgb_black = [0, 0, 0]
    rgb_red = [255, 0, 0]
    atom_pattern = re.compile('[A-Za-z]')

    @staticmethod
    def get_id():
        return 'substructure_highlight'

    @staticmethod
    def get_name():
        return 'Substructure Highlight'

    @staticmethod
    def get_parameters():
        parameters = list()
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_target(global_parameters)

    @staticmethod
    def execute(global_parameters, local_parameters):
        data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
        smiles = data_h5[file_structure.DataSet.smiles]
        output_dir_path = file_util.resolve_subpath(file_structure.get_interpretation_folder(global_parameters),
                                                    'substructure_highlight')
        file_util.make_folders(output_dir_path, True)
        substructures = hdf5_util.get_property(file_structure.get_target_file(global_parameters),
                                               'substructures').split(';')
        for i in range(len(substructures)):
            substructures[i] = Chem.MolFromSmiles(substructures[i], sanitize=False)
        with progressbar.ProgressBar(len(smiles)) as progress:
            for i in range(len(smiles)):
                output_path = file_util.resolve_subpath(output_dir_path, str(i) + '.svg')
                if not file_util.file_exists(output_path):
                    smiles_string = smiles[i].decode('utf-8')
                    molecule = Chem.MolFromSmiles(smiles_string, sanitize=False)
                    indices = set()
                    for substructure in substructures:
                        matches = molecule.GetSubstructMatches(substructure)
                        for match in matches:
                            for index in match:
                                indices.add(index)
                    heatmap = SubstructureHighlight.generate_heatmap(smiles_string, indices)
                    smiles_renderer.render(smiles_string, output_path, 5, heatmap)
                progress.increment()
        data_h5.close()

    @staticmethod
    def generate_heatmap(smiles_string, indices):
        heatmap = list()
        index = 0
        for i in range(len(smiles_string)):
            color = SubstructureHighlight.rgb_black
            if SubstructureHighlight.atom_pattern.match(smiles_string[i]):
                if index in indices:
                    color = SubstructureHighlight.rgb_red
                index += 1
            heatmap.append(color)
        return heatmap
