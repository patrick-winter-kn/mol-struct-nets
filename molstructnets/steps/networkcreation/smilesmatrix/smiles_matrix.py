class SmilesMatrix:

    @staticmethod
    def get_id():
        return 'smiles_matrix'

    @staticmethod
    def get_name():
        return 'SMILES Matrix'

    @staticmethod
    def get_parameters():
        parameters = list()
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        dimensions = global_parameters['input_dimensions']
        if len(dimensions) != 2:
            raise ValueError('Preprocessed dimensions are not 1D')

    @staticmethod
    def execute(global_parameters, parameters):
        # TODO
        raise NotImplementedError('This method has not yet been implemented')
