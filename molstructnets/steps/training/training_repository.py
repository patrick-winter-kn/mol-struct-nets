from steps import repository
from steps.training.image import image
from steps.training.randomforest import random_forest
from steps.training.tensor import tensor
from steps.training.tensor2djit import tensor_2d_jit


class TrainingRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'training'

    @staticmethod
    def get_name():
        return 'Training'


instance = TrainingRepository()
instance.add_implementation(tensor.Tensor)
instance.add_implementation(tensor_2d_jit.Tensor2DJit)
instance.add_implementation(image.Image)
instance.add_implementation(random_forest.RandomForest)
