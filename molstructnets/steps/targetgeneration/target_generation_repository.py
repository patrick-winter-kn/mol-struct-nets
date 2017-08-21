from steps import repository


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'target_generation'

    @staticmethod
    def get_name():
        return 'Target Generation'


instance = DataGenerationRepository()
