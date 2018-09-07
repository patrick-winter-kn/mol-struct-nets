from steps import repository
from steps.interpretation.calculate2dsubstructurelocations import calculate_2d_substructure_locations
from steps.interpretation.calculatecams2d import calculate_cams_2d
from steps.interpretation.camevaluation2d import cam_evaluation_2d
from steps.interpretation.extractcamsubstructures2d import extract_cam_substructures_2d
from steps.interpretation.rendercams2d import render_cams_2d
from steps.interpretation.rendersubstructurelocations2d import render_substructure_locations_2d


class InterpretationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'interpretation'

    @staticmethod
    def get_name():
        return 'Interpretation'


instance = InterpretationRepository()
instance.add_implementation(calculate_cams_2d.CalculateCams2D)
instance.add_implementation(render_cams_2d.RenderCams2D)
instance.add_implementation(extract_cam_substructures_2d.ExtractCamSubstructures2D)
instance.add_implementation(calculate_2d_substructure_locations.Calculate2DSubstructureLocations)
instance.add_implementation(render_substructure_locations_2d.RenderSubstructureLocations2D)
instance.add_implementation(cam_evaluation_2d.CamEvaluation2D)
