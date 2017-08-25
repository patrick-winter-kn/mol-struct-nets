from steps import repository
from steps.partition.stratifiedsampling import stratified_sampling


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'partition'

    @staticmethod
    def get_name():
        return 'Partition'


instance = DataGenerationRepository()
instance.add_implementation(stratified_sampling.StratifiedSampling)
