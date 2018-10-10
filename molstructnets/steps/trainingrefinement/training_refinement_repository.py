from steps import repository
from steps.trainingrefinement.tensor2d import tensor_2d


class TrainingRefinementRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'training_refinement'

    @staticmethod
    def get_name():
        return 'Training (Refinement)'


instance = TrainingRefinementRepository()
instance.add_implementation(tensor_2d.Tensor2D)
