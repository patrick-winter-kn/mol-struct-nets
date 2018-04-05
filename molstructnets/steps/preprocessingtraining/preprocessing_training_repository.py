from steps import repository
from steps.preprocessingtraining.tensorsmilestransformation import tensor_smiles_transformation
from steps.preprocessingtraining.tensor2dtransformation import tensor_2d_transformation


class PreprocessingTrainingRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'preprocessing_training'

    @staticmethod
    def get_name():
        return 'Preprocessing (Training)'


instance = PreprocessingTrainingRepository()
instance.add_implementation(tensor_smiles_transformation.TensorSmilesTransformation)
instance.add_implementation(tensor_2d_transformation.Tensor2DTransformed)
