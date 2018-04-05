from steps import repository
from steps.targetgeneration.substructure import substructure


class TargetGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'target_generation'

    @staticmethod
    def get_name():
        return 'Target Generation'


instance = TargetGenerationRepository()
instance.add_implementation(substructure.Substructure)
