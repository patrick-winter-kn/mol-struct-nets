from steps import repository
from steps.evaluation.enrichmentplot import enrichment_plot
from steps.evaluation.roccurveplot import roc_curve_plot
from steps.evaluation.riebedroc import rie_bedroc


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'evaluation'

    @staticmethod
    def get_name():
        return 'Evaluation'


instance = DataGenerationRepository()
instance.add_implementation(enrichment_plot.EnrichmentPlot)
instance.add_implementation(roc_curve_plot.RocCurvePlot)
instance.add_implementation(rie_bedroc.RieBedroc)
