from steps import repository
from steps.training.randomforest import random_forest
from steps.training.tensor2d import tensor_2d


class TrainingRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'training'

    @staticmethod
    def get_name():
        return 'Training'


instance = TrainingRepository()
instance.add_implementation(tensor_2d.Tensor2D)
instance.add_implementation(random_forest.RandomForest)
