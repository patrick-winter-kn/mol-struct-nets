from steps import repository
from steps.partitioning.stratifiedsampling import stratified_sampling


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'partitioning'

    @staticmethod
    def get_name():
        return 'Partitioning'


instance = DataGenerationRepository()
instance.add_implementation(stratified_sampling.StratifiedSampling)
