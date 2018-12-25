from steps import repository
from steps.featuregeneration.ecfpfingerprint import ecfp_fingerprint
from steps.featuregeneration.learnedfeaturegenerationtensor2d import learned_feature_generation_tensor_2d
from steps.featuregeneration.maccsfingerprint import maccs_fingerprint
from steps.featuregeneration.saliencymapsubstructurefeaturegeneration import saliency_map_substructure_feature_generation
from steps.featuregeneration.mossfeaturegeneration import moss_feature_generation


class FeatureGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'feature_generation'

    @staticmethod
    def get_name():
        return 'Feature Generation'


instance = FeatureGenerationRepository()
instance.add_implementation(learned_feature_generation_tensor_2d.LearnedFeatureGenerationTensor2D)
instance.add_implementation(ecfp_fingerprint.EcfpFingerprint)
instance.add_implementation(maccs_fingerprint.MaccsFingerprint)
instance.add_implementation(saliency_map_substructure_feature_generation.SaliencyMapSubstructureFeatureGeneration)
instance.add_implementation(moss_feature_generation.MossFeatureGeneration)
