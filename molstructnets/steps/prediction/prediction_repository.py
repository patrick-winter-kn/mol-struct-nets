from steps import repository
from steps.prediction.matrix import matrix
from steps.prediction.image import image


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'prediction'

    @staticmethod
    def get_name():
        return 'Prediction'


instance = DataGenerationRepository()
instance.add_implementation(matrix.Matrix)
instance.add_implementation(image.Image)
