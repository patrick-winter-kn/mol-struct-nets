from steps import repository
from steps.networkcreation.tensorsmiles import tensor_smiles
from steps.networkcreation.tensor2d import tensor_2d
from steps.networkcreation.image import image
from steps.networkcreation.vgg19 import vgg19
from steps.networkcreation.customtensor2d import custom_tensor_2d
from steps.networkcreation.mlp import mlp
from steps.networkcreation.tensor2dadaptive import tensor_2d_adaptive


class NetworkCreationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'network_creation'

    @staticmethod
    def get_name():
        return 'Network Creation'


instance = NetworkCreationRepository()
instance.add_implementation(mlp.MLP)
instance.add_implementation(tensor_smiles.TensorSmiles)
instance.add_implementation(tensor_2d.Tensor2D)
instance.add_implementation(tensor_2d_adaptive.Tensor2DAdaptive)
instance.add_implementation(custom_tensor_2d.CustomTensor2D)
instance.add_implementation(image.Image)
instance.add_implementation(vgg19.Vgg19)
