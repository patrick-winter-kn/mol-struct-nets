from steps import repository


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'data_generation'

    @staticmethod
    def get_name():
        return 'Data Generation'


instance = DataGenerationRepository()
