from steps import repository
from steps.preprocessing.smilesmatrix import smiles_matrix
from steps.preprocessing.image import image
from steps.preprocessing.matrix2d import matrix_2d
from steps.preprocessing.fingerprint import fingerprint


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'preprocessing'

    @staticmethod
    def get_name():
        return 'Preprocessing'


instance = DataGenerationRepository()
instance.add_implementation(smiles_matrix.SmilesMatrix)
instance.add_implementation(image.Image)
instance.add_implementation(matrix_2d.Matrix2D)
instance.add_implementation(fingerprint.Fingerprint)
