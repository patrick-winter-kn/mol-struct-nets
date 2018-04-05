from steps import repository
from steps.prediction.tensor import tensor
from steps.prediction.image import image
from steps.prediction.randomforest import random_forest


class PredictionRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'prediction'

    @staticmethod
    def get_name():
        return 'Prediction'


instance = PredictionRepository()
instance.add_implementation(tensor.Tensor)
instance.add_implementation(image.Image)
instance.add_implementation(random_forest.RandomForest)
