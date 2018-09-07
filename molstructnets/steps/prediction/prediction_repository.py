from steps import repository
from steps.prediction.randomforest import random_forest
from steps.prediction.tensor2d import tensor_2d


class PredictionRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'prediction'

    @staticmethod
    def get_name():
        return 'Prediction'


instance = PredictionRepository()
instance.add_implementation(tensor_2d.Tensor2D)
instance.add_implementation(random_forest.RandomForest)
