from steps import repository
from steps.preprocessing.tensorsmiles import tensor_smiles
from steps.preprocessing.image import image
from steps.preprocessing.tensor2d import tensor_2d
from steps.preprocessing.tensor2djit import tensor_2d_jit


class PreprocessingRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'preprocessing'

    @staticmethod
    def get_name():
        return 'Preprocessing'


instance = PreprocessingRepository()
instance.add_implementation(tensor_smiles.TensorSmiles)
instance.add_implementation(tensor_2d.Tensor2D)
instance.add_implementation(tensor_2d_jit.Tensor2DJit)
instance.add_implementation(image.Image)
