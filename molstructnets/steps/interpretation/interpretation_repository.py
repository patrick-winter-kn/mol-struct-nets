from steps import repository
from steps.interpretation.calculate2dsubstructurelocations import calculate_2d_substructure_locations
from steps.interpretation.calculatesaliencymaps2d import calculate_saliency_maps_2d
from steps.interpretation.saliencymapevaluation2d import saliency_map_evaluation_2d
from steps.interpretation.extractsaliencymapsubstructures2d import extract_saliency_map_substructures_2d
from steps.interpretation.rendersaliencymaps2d import render_saliency_maps_2d
from steps.interpretation.rendersubstructurelocations2d import render_substructure_locations_2d


class InterpretationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'interpretation'

    @staticmethod
    def get_name():
        return 'Interpretation'


instance = InterpretationRepository()
instance.add_implementation(calculate_saliency_maps_2d.CalculateSaliencyMaps2D)
instance.add_implementation(render_saliency_maps_2d.RenderSaliencyMaps2D)
instance.add_implementation(extract_saliency_map_substructures_2d.ExtractSaliencyMapSubstructures2D)
instance.add_implementation(calculate_2d_substructure_locations.Calculate2DSubstructureLocations)
instance.add_implementation(render_substructure_locations_2d.RenderSubstructureLocations2D)
instance.add_implementation(saliency_map_evaluation_2d.SaliencyMapEvaluation2D)
