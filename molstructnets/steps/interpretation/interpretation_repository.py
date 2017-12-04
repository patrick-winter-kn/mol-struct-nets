from steps import repository
from steps.interpretation.calculateattentionmaps import calculate_attention_maps
from steps.interpretation.calculatesmilessubstructureatoms import calculate_smiles_substructure_atoms
from steps.interpretation.calculate2dsubstructureatoms import calculate_2d_substructure_atoms
from steps.interpretation.renderattentionmaps import render_attention_maps
from steps.interpretation.rendersubstructureatoms import render_substructure_atoms
from steps.interpretation.attentionevaluation import attention_evaluation
from steps.interpretation.smilesattentionsubstructures import smiles_attention_substructures


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'interpretation'

    @staticmethod
    def get_name():
        return 'Interpretation'


instance = DataGenerationRepository()
instance.add_implementation(calculate_attention_maps.CalculateAttentionMaps)
instance.add_implementation(calculate_smiles_substructure_atoms.CalculateSmilesSubstructureAtoms)
instance.add_implementation(calculate_2d_substructure_atoms.Calculate2DSubstructureAtoms)
instance.add_implementation(render_attention_maps.RenderAttentionMaps)
instance.add_implementation(render_substructure_atoms.RenderSubstructureAtoms)
instance.add_implementation(attention_evaluation.AttentionEvaluation)
instance.add_implementation(smiles_attention_substructures.SmilesAttentionSubstructures)
