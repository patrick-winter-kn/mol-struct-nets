from steps import repository


class TrainingMultitargetRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'training_multitarget'

    @staticmethod
    def get_name():
        return 'Training (Multitarget)'


instance = TrainingMultitargetRepository()
