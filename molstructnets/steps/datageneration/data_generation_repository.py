from steps import repository
from steps.datageneration.randomsmiles import random_smiles


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'data_generation'

    @staticmethod
    def get_name():
        return 'Data Generation'


instance = DataGenerationRepository()
instance.add_implementation(random_smiles.RandomSmiles)
