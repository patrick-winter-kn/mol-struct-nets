from steps import repository
from steps.datageneration import data_generation_repository
from steps.targetgeneration import target_generation_repository
from steps.partitioning import partitioning_repository
from steps.preprocessing import preprocessing_repository
from steps.preprocessingtraining import preprocessing_training_repository
from steps.featuregeneration import feature_generation_repository
from steps.networkcreation import network_creation_repository
from steps.training import training_repository
from steps.prediction import prediction_repository
from steps.evaluation import evaluation_repository
from steps.interpretation import interpretation_repository


class StepsRepository(repository.Repository):

    def get_steps(self):
        return self.get_implementations()

    def get_step_names(self):
        names = []
        for implementation in self.get_implementations():
            names.append(implementation.get_name())
        return names

    def get_step_name(self, type_):
        return self.get_implementation(type_).get_name()

    def get_step_implementations(self, type_):
        return self.get_implementation(type_).get_implementations()

    def get_step_implementation_names(self, type_):
        names = []
        for implementation in self.get_step_implementations(type_):
            names.append(implementation.get_name())
        return names

    def get_step_implementation(self, type_, id_):
        return self.get_implementation(type_).get_implementation(id_)


instance = StepsRepository()
instance.add_implementation(data_generation_repository.instance)
instance.add_implementation(target_generation_repository.instance)
instance.add_implementation(partitioning_repository.instance)
instance.add_implementation(preprocessing_repository.instance)
instance.add_implementation(preprocessing_training_repository.instance)
instance.add_implementation(feature_generation_repository.instance)
instance.add_implementation(network_creation_repository.instance)
instance.add_implementation(training_repository.instance)
instance.add_implementation(prediction_repository.instance)
instance.add_implementation(evaluation_repository.instance)
instance.add_implementation(interpretation_repository.instance)
