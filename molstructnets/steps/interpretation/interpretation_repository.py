from steps import repository
from steps.interpretation.calculatesmilesattention import calculate_smiles_attention
from steps.interpretation.calculatesubstructureatoms import calculate_substructure_atoms
from steps.interpretation.rendersmilesattention import render_smiles_attention
from steps.interpretation.rendersubstructureatoms import render_substructure_atoms


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'interpretation'

    @staticmethod
    def get_name():
        return 'Interpretation'


instance = DataGenerationRepository()
instance.add_implementation(calculate_smiles_attention.CalculateSmilesAttention)
instance.add_implementation(calculate_substructure_atoms.CalculateSubstructureAtoms)
instance.add_implementation(render_smiles_attention.RenderSmilesAttention)
instance.add_implementation(render_substructure_atoms.RenderSubstructureAtoms)
