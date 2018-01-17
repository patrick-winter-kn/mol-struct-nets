from steps import repository
from steps.training.matrix import matrix
from steps.training.image import image
from steps.training.randomforest import random_forest


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'training'

    @staticmethod
    def get_name():
        return 'Training'


instance = DataGenerationRepository()
instance.add_implementation(matrix.Matrix)
instance.add_implementation(image.Image)
instance.add_implementation(random_forest.RandomForest)
