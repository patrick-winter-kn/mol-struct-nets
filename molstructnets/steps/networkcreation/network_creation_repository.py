from steps import repository
from steps.networkcreation.customtensor2d import custom_tensor_2d
from steps.networkcreation.tensor2d import tensor_2d
from steps.networkcreation.tensor2dadaptive import tensor_2d_adaptive


class NetworkCreationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'network_creation'

    @staticmethod
    def get_name():
        return 'Network Creation'


instance = NetworkCreationRepository()
instance.add_implementation(tensor_2d.Tensor2D)
instance.add_implementation(tensor_2d_adaptive.Tensor2DAdaptive)
instance.add_implementation(custom_tensor_2d.CustomTensor2D)
