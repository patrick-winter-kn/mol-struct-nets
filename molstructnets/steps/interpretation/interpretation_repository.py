from steps import repository
from steps.interpretation.smilesattention import smiles_attention


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'interpretation'

    @staticmethod
    def get_name():
        return 'Interpretation'


instance = DataGenerationRepository()
instance.add_implementation(smiles_attention.SmilesAttention)
