from steps import repository


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'training'

    @staticmethod
    def get_name():
        return 'Training'


instance = DataGenerationRepository()
