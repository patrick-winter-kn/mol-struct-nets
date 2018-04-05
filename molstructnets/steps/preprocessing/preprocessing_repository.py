from steps import repository
from steps.preprocessing.tensorsmiles import tensor_smiles
from steps.preprocessing.image import image
from steps.preprocessing.tensor2d import tensor_2d
from steps.preprocessing.ecfpfingerprint import ecfp_fingerprint
from steps.preprocessing.maccsfingerprint import maccs_fingerprint
from steps.preprocessing.learnedfeaturegeneration import learned_feature_generation


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'preprocessing'

    @staticmethod
    def get_name():
        return 'Preprocessing'


instance = DataGenerationRepository()
instance.add_implementation(tensor_smiles.TensorSmiles)
instance.add_implementation(tensor_2d.Tensor2D)
instance.add_implementation(learned_feature_generation)
instance.add_implementation(image.Image)
instance.add_implementation(ecfp_fingerprint.EcfpFingerprint)
instance.add_implementation(maccs_fingerprint.MaccsFingerprint)
