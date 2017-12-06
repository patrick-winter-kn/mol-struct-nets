from steps import repository
from steps.networkcreation.smilesmatrix import smiles_matrix
from steps.networkcreation.matrix2d import matrix_2d
from steps.networkcreation.image import image
from steps.networkcreation.vgg19 import vgg19
from steps.networkcreation.custommatrix2d import custom_matrix_2d


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'network_creation'

    @staticmethod
    def get_name():
        return 'Network Creation'


instance = DataGenerationRepository()
instance.add_implementation(smiles_matrix.SmilesMatrix)
instance.add_implementation(matrix_2d.Matrix2D)
instance.add_implementation(image.Image)
instance.add_implementation(vgg19.Vgg19)
instance.add_implementation(custom_matrix_2d.CustomMatrix2D)
