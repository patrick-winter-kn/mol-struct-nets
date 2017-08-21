from steps import repository


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'evaluation'

    @staticmethod
    def get_name():
        return 'Evaluation'


instance = DataGenerationRepository()
