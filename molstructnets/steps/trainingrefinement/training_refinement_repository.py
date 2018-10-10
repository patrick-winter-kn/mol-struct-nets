from steps import repository


class TrainingRefinementRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'training_refinement'

    @staticmethod
    def get_name():
        return 'Training (Refinement)'


instance = TrainingRefinementRepository()
