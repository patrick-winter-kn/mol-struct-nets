import re

import h5py

from steps.interpretation.shared import smiles_renderer
from util import data_validation, file_structure, file_util, progressbar, logger


class RenderSubstructureAtoms:

    rgb_black = [0, 0, 0]
    rgb_red = [255, 0, 0]
    atom_pattern = re.compile('[A-Za-z]')

    @staticmethod
    def get_id():
        return 'render_substructure_atoms'

    @staticmethod
    def get_name():
        return 'Render Substructure Atoms'

    @staticmethod
    def get_parameters():
        parameters = list()
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, local_parameters):
        data_validation.validate_data_set(global_parameters)
        data_validation.validate_attention_map(global_parameters, file_structure.AttentionMap.substructure_atoms)

    @staticmethod
    def execute(global_parameters, local_parameters):
        attention_map_h5 = h5py.File(file_structure.get_attentionmap_file(global_parameters), 'r')
        data_h5 = h5py.File(file_structure.get_data_set_file(global_parameters), 'r')
        smiles = data_h5[file_structure.DataSet.smiles]
        if file_structure.AttentionMap.substructure_atoms in attention_map_h5.keys():
            substructure_atoms_dir_path = file_util.resolve_subpath(file_structure.get_interpretation_folder(global_parameters),
                                                        'substructure_atoms')
            file_util.make_folders(substructure_atoms_dir_path, True)
            substructure_atoms = attention_map_h5[file_structure.AttentionMap.substructure_atoms]
            logger.log('Rendering substructure atoms', logger.LogLevel.INFO)
            RenderSubstructureAtoms.render(substructure_atoms, smiles, substructure_atoms_dir_path)
        attention_map_h5.close()
        data_h5.close()

    @staticmethod
    def render(substructure_atoms, smiles, output_dir_path):
        with progressbar.ProgressBar(len(smiles)) as progress:
            for i in range(len(smiles)):
                output_path = file_util.resolve_subpath(output_dir_path, str(i) + '.svg')
                if not file_util.file_exists(output_path):
                    smiles_string = smiles[i].decode('utf-8')
                    heatmap = RenderSubstructureAtoms.generate_heatmap(substructure_atoms[i])
                    smiles_renderer.render(smiles_string, output_path, 5, heatmap)
                progress.increment()

    @staticmethod
    def generate_heatmap(substructure_atoms):
        heatmap = list()
        for i in range(len(substructure_atoms)):
            if substructure_atoms[i] == 1:
                heatmap.append(RenderSubstructureAtoms.rgb_red)
            else:
                heatmap.append(RenderSubstructureAtoms.rgb_black)
        return heatmap
