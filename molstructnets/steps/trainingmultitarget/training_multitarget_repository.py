from steps import repository
from steps.trainingmultitarget.tensor2d import tensor_2d


class TrainingMultitargetRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'training_multitarget'

    @staticmethod
    def get_name():
        return 'Training (Multitarget)'


instance = TrainingMultitargetRepository()
instance.add_implementation(tensor_2d.Tensor2D)
