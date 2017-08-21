from steps import repository


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'partition'

    @staticmethod
    def get_name():
        return 'Partition'


instance = DataGenerationRepository()
