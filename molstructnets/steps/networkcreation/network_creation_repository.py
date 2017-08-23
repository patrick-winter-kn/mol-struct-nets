from steps import repository
from steps.networkcreation.smilesmatrix import smiles_matrix
from steps.networkcreation.image import image


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'network_creation'

    @staticmethod
    def get_name():
        return 'Network Creation'


instance = DataGenerationRepository()
instance.add_implementation(smiles_matrix.SmilesMatrix)
instance.add_implementation(image.Image)
