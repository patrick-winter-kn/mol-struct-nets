from steps import repository
from steps.featuregeneration.ecfpfingerprint import ecfp_fingerprint
from steps.featuregeneration.learnedfeaturegeneration import learned_feature_generation
from steps.featuregeneration.learnedfeaturegenerationtensor2djit import learned_feature_generation_tensor_2d_jit
from steps.featuregeneration.maccsfingerprint import maccs_fingerprint
from steps.featuregeneration.camsubstructurefeaturegeneration import cam_substructure_feature_generation


class FeatureGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'feature_generation'

    @staticmethod
    def get_name():
        return 'Feature Generation'


instance = FeatureGenerationRepository()
instance.add_implementation(learned_feature_generation.LearnedFeatureGeneration)
instance.add_implementation(learned_feature_generation_tensor_2d_jit.LearnedFeatureGenerationTensor2DJit)
instance.add_implementation(ecfp_fingerprint.EcfpFingerprint)
instance.add_implementation(maccs_fingerprint.MaccsFingerprint)
instance.add_implementation(cam_substructure_feature_generation.CamSubstructureFeatureGeneration)
