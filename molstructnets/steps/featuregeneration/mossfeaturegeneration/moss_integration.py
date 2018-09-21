import subprocess
import sys

from rdkit import Chem
from util import file_util


def calculate_substructures(smiles_data, classes, min_focus_support, max_complement_support, active):
    input_file_path = file_util.get_temporary_file_path('moss_input')
    output_file_path = file_util.get_temporary_file_path('moss_output')
    if active:
        class_index = 1
    else:
        class_index = 0
    with open(input_file_path, 'w') as input_file:
        for i in range(len(smiles_data)):
            input_file.write(str(i) + ',' + str(classes[i, class_index]) + ',' + smiles_data[i].decode('UTF-8') + '\n')
    moss_jar = [sys.argv[0][:sys.argv[0].rfind('/') + 1] +
                'steps/featuregeneration/mossfeaturegeneration/moss/moss.jar']
    params = ['java', '-cp'] + moss_jar + ['moss.Miner', input_file_path, output_file_path]
    subprocess.call(params)
    # separator after MoSS output
    print('')
    substructures = list()
    with open(output_file_path, 'r') as output_file:
        # ignore headers line
        output_file.readline()
        line = output_file.readline()
        while line:
            # remove newline before split
            values = line[:-1].split(',')
            smiles = values[1]
            focus_support = float(values[5])
            complement_support = float(values[7])
            if focus_support >= min_focus_support and complement_support <= max_complement_support:
                substructures.append(Chem.MolFromSmiles(smiles, sanitize=False))
            line = output_file.readline()
    return substructures
