from steps import repository
from steps.preprocessing.tensor2d import tensor_2d


class PreprocessingRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'preprocessing'

    @staticmethod
    def get_name():
        return 'Preprocessing'


instance = PreprocessingRepository()
instance.add_implementation(tensor_2d.Tensor2D)
