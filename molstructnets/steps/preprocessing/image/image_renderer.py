from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import DrawingOptions

from util import file_util


class ImageRenderer:

    def __init__(self, directory, progress, size):
        self.directory = directory
        self.progress = progress
        self.size = size

    def render(self, smiles_data, offset):
        scale = self.size / 800
        drawing_size = (self.size, self.size)
        options = DrawingOptions()
        options.dotsPerAngstrom = 30 * scale
        for i in range(len(smiles_data)):
            image_path = file_util.resolve_subpath(self.directory, str(offset + i) + '.png')
            if not file_util.file_exists(image_path):
                molecule = Chem.MolFromSmiles(smiles_data[i], sanitize=False)
                Draw.MolToFile(molecule, image_path, size=drawing_size, options=options)
            self.progress.increment()
