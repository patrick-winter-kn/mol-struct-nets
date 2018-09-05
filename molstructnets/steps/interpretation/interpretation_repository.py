from steps import repository
from steps.interpretation.calculate2dsubstructurelocationsjit import calculate_2d_substructure_locations_jit
from steps.interpretation.calculatecams2djit import calculate_cams_2d_jit
from steps.interpretation.camevaluation2djit import cam_evaluation_2d_jit
from steps.interpretation.extractcamsubstructures2djit import extract_cam_substructures_2d_jit
from steps.interpretation.rendercams2djit import render_cams_2d_jit
from steps.interpretation.rendersubstructurelocations2djit import render_substructure_locations_2d_jit


class InterpretationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'interpretation'

    @staticmethod
    def get_name():
        return 'Interpretation'


instance = InterpretationRepository()
instance.add_implementation(calculate_cams_2d_jit.CalculateCams2DJit)
instance.add_implementation(render_cams_2d_jit.RenderCams2DJit)
instance.add_implementation(extract_cam_substructures_2d_jit.ExtractCamSubstructures2DJit)
instance.add_implementation(calculate_2d_substructure_locations_jit.Calculate2DSubstructureLocationsJit)
instance.add_implementation(render_substructure_locations_2d_jit.RenderSubstructureLocations2DJit)
instance.add_implementation(cam_evaluation_2d_jit.CamEvaluation2DJit)
