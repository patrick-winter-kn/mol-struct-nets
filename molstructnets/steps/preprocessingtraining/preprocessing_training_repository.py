from steps import repository
from steps.preprocessingtraining.smilesmatrixtransformation import smiles_matrix_transformation
from steps.preprocessingtraining.matrix2dtransformation import matrix_2d_transformation


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'preprocessing_training'

    @staticmethod
    def get_name():
        return 'Preprocessing (Training)'


instance = DataGenerationRepository()
instance.add_implementation(smiles_matrix_transformation.SmilesMatrixTransformation)
instance.add_implementation(matrix_2d_transformation.Matrix2DTransformed)
