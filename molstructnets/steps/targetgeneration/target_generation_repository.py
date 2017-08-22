from steps import repository
from steps.targetgeneration.substructure import substructure


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'target_generation'

    @staticmethod
    def get_name():
        return 'Target Generation'


instance = DataGenerationRepository()
instance.add_implementation(substructure.Substructure)
