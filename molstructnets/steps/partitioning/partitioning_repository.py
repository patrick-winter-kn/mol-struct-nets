from steps import repository
from steps.partitioning.stratifiedsampling import stratified_sampling
from steps.partitioning.postprocessing import postprocessing


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'partitioning'

    @staticmethod
    def get_name():
        return 'Partitioning'


instance = DataGenerationRepository()
instance.add_implementation(stratified_sampling.StratifiedSampling)
instance.add_implementation(postprocessing.Postprocessing)
