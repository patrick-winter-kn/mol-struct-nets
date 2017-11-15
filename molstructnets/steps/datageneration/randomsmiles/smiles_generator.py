import random
from util import misc
import re
from rdkit import Chem, RDLogger


# Available SMILES string elements
class Elements:
    atom = 'atom'
    branch = 'branch'
    ring = 'ring'


# Probability in percent that this atom occurs
atom_probabilities = {'O': 0.13, 'S': 0.03, 'N': 0.12, 'C': 0.72}
# Number of bonds that this atom can have
atom_valencies = {'O': 2, 'S': 6, 'N': 3, 'C': 4}
# Probability in percent taht this element occurs
element_probabilities = {Elements.atom: 0.67, Elements.branch: 0.19, Elements.ring: 0.14}
# Mattern that matches atoms
atom_pattern = re.compile('[A-Z]')


class SmilesGenerator:

    def __init__(self, number, max_length, seed, offset=0, progress=None, global_smiles_set=None):
        self.number = number
        self.max_length = max_length
        self.random = random.Random()
        self.seed = seed
        self.offset = offset
        self.progress = progress
        self.global_smiles_set = global_smiles_set
        self.ring_label = 1

    def pick_option(self, option_probabilities_map):
        if len(option_probabilities_map) == 0:
            # No options available, return None
            return None
        # Example for options {'A':2, 'B':5, 'C':3}
        # |==|=====|===|
        # A  B     C
        #        ^
        # Pick random number in p(A)+p(B)+p(C)
        # Substract probabilities until rest is smaller equals zero to find area that the picked number belongs to
        max_value = sum(option_probabilities_map.values())
        rest = self.random.uniform(0.0, max_value)
        for option in sorted(option_probabilities_map.keys()):
            rest -= option_probabilities_map[option]
            if rest <= 0:
                return option

    @staticmethod
    def atoms_with_min_valency(min_valency):
        atoms = set()
        # Add all atoms with at least min_valency to new set
        for atom in atom_valencies:
            if atom_valencies[atom] >= min_valency:
                atoms.add(atom)
        return atoms

    @staticmethod
    def branch_possible(remaining_length, previous_atom, used_bonds):
        if previous_atom is None:
            # Can't start with branch
            return False
        if remaining_length < 1:
            # Branch can't be empty
            return False
        if atom_valencies[previous_atom] - used_bonds < 2:
            # Atom needs to at least have space for 2 more bonds (this branch and the following mainline)
            return False
        # We found no issue
        return True

    @staticmethod
    def ring_possible(remaining_length, previous_atom, used_bonds):
        if previous_atom is None:
            # Ring can't start without a first atom
            return False
        if remaining_length < 2:
            # A ring needs at least 3 atoms (start atom and 2 inbetween labels)
            return False
        if atom_valencies[previous_atom] - used_bonds < 2:
            # Atom needs to at least have space for 2 more bonds (start of ring and end of ring)
            return False
        # We found no issue
        return True

    def generate_smiles(self, length, bonds_in_front, bonds_in_back, in_ring):
        string = ''
        # The atom that the next element needs to be bonded to
        previous_atom = None
        # The number of bonds that are already used for the previous atom
        used_bonds = bonds_in_front
        # Ring mainline checks that at least two atoms are present inside the ring (not counting branches)
        ring_mainline = 0
        # Add elements until the string has the desired length
        while len(string) < length:
            remaining_length = length - len(string)
            # Set of elements that are an option under the given circumstances (atoms are always an option)
            valid_elements = {Elements.atom}
            # A branch needs space for the starting and ending parenthesizes
            remaining_branch_length = remaining_length - 2
            # If the previous atom does not support the currently used bonds and the bonds after this substring we have
            # to save space for another atom after this branch
            if previous_atom is not None and atom_valencies[previous_atom] - used_bonds - bonds_in_back < 1:
                remaining_branch_length -= 1
            # If we are in a ring and do not have enough atoms in the mainline we need to keep space for another
            elif in_ring and ring_mainline < 2:
                remaining_branch_length -= 1
            ring_label = str(self.ring_label)
            # Labels with more than one character (>9) need to start with a %
            if len(ring_label) > 1:
                ring_label = '%' + ring_label
            # A ring needs space for the label at the start and the end
            remaining_ring_length = remaining_length - 2 * len(ring_label)
            if max(atom_valencies.values()) - bonds_in_back < 2:
                # There is no atom that can support another ring being closed so this ring can't go to the end
                remaining_ring_length -= 1
            # Check if we can add a branch
            if SmilesGenerator.branch_possible(remaining_branch_length, previous_atom, used_bonds):
                valid_elements.add(Elements.branch)
            # Check if we can add a ring
            if SmilesGenerator.ring_possible(remaining_ring_length, previous_atom, used_bonds):
                valid_elements.add(Elements.ring)
            # Pick element to add
            element = self.pick_option(misc.copy_dict_from_keys(element_probabilities, valid_elements))
            if element == Elements.atom:
                if len(string) == 0 and bonds_in_front == 0:
                    # We are at the beginning of the entire SMILES and need no bond in the front
                    used_bonds = 0
                else:
                    # One bond to previous atom
                    used_bonds = 1
                min_valency = used_bonds
                if remaining_length > 1:
                    # We are not at the end of the SMILES string and need at least one bond for a following element
                    min_valency += 1
                else:
                    # We are at the end of this substring and need bonds for the elements in the back
                    min_valency += bonds_in_back
                # Pick atom that has enough bonds
                atom = self.pick_option(misc.copy_dict_from_keys(atom_probabilities,
                                                                 SmilesGenerator.atoms_with_min_valency(min_valency)))
                # Add atom to SMILES string
                string += atom
                # Remember atom
                previous_atom = atom
                # We added an atom to the ring
                ring_mainline += 1
            elif element == Elements.branch:
                # The branch is connected with one bond
                used_bonds += 1
                # Decide branch length
                inner_length = self.random.randint(1, remaining_branch_length)
                # Generate branch content
                branch = self.generate_smiles(inner_length, 1, 0, False)[0]
                # Add branch to SMILES string
                string += '(' + branch + ')'
            elif element == Elements.ring:
                # The current ring label will be used increment the global one
                self.ring_label += 1
                # Decide ring length
                inner_length = self.random.randint(2, remaining_ring_length)
                # The inner content needs to connect back to the previous atom to form a ring
                inner_bonds_in_back = 1
                if inner_length + 2 * len(ring_label) < remaining_length:
                    # The inner content also needs to connect to the next element after the ring
                    inner_bonds_in_back += 1
                else:
                    # The inner content also needs to connect to what ever comes behind this substring
                    inner_bonds_in_back += bonds_in_back
                # Generate ring content
                ring, previous_atom, used_bonds = self.generate_smiles(inner_length, 1, inner_bonds_in_back, True)
                # Add ring content to SMILES string
                string += ring_label + ring + ring_label
                # We added at least to atoms to the ring
                ring_mainline += 2
        return string, previous_atom, used_bonds + bonds_in_back

    def write_smiles(self, array):
        # Turn of error logging of RDKit
        logger = RDLogger.logger()
        logger.setLevel(RDLogger.CRITICAL)
        # Iterate over assigned area of array
        for i in range(self.offset, self.number + self.offset):
            # Set seed based on general seed and position in array
            self.random.seed(self.seed + i)
            smiles = None
            valid = False
            # Run until we found a valid SMILES for this position
            while not valid:
                # Reset ring label
                self.ring_label = 1
                # Decide length
                length = self.random.randint(1, self.max_length)
                # Generate SMILES
                try:
                    smiles = self.generate_smiles(length, 0, 0, False)[0]
                except Exception:
                    # If something goes wrong try again
                    continue
                # Parse with RDKit
                mol = Chem.MolFromSmiles(smiles)
                # If it could not be parsed or failed a sanity check it is None
                if mol is None:
                    # Rerun loop and try again
                    continue
                # Get SMILES back (in canonical form)
                smiles = Chem.MolToSmiles(mol)
                # Check if canonical SMILES does not exceed maximum length
                if len(smiles) <= self.max_length:
                    # Check if SMILES has not been generated before
                    valid = self.global_smiles_set is None or self.global_smiles_set.add(smiles)
            array[i] = smiles.encode('utf-8')
            self.progress.increment()
        # Turn error logging of RDKit back on
        logger.setLevel(RDLogger.DEBUG)
