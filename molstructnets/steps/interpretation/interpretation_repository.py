from steps import repository
from steps.interpretation.calculatecams import calculate_cams
from steps.interpretation.calculatesmilessubstructureatoms import calculate_smiles_substructure_atoms
from steps.interpretation.calculate2dsubstructureatoms import calculate_2d_substructure_atoms
from steps.interpretation.rendercams import render_cams
from steps.interpretation.rendersubstructureatoms import render_substructure_atoms
from steps.interpretation.camevaluation import cam_evaluation
from steps.interpretation.extractcamsubstructures import extract_cam_substructures


class DataGenerationRepository(repository.Repository):

    @staticmethod
    def get_id():
        return 'interpretation'

    @staticmethod
    def get_name():
        return 'Interpretation'


instance = DataGenerationRepository()
instance.add_implementation(calculate_cams.CalculateCams)
instance.add_implementation(calculate_smiles_substructure_atoms.CalculateSmilesSubstructureAtoms)
instance.add_implementation(calculate_2d_substructure_atoms.Calculate2DSubstructureAtoms)
instance.add_implementation(render_cams.RenderCams)
instance.add_implementation(render_substructure_atoms.RenderSubstructureAtoms)
instance.add_implementation(cam_evaluation.CamEvaluation)
instance.add_implementation(extract_cam_substructures.ExtractCamSubstructures)
