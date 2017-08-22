class RandomSmiles:

    @staticmethod
    def get_id():
        return 'random_smiles'

    @staticmethod
    def get_name():
        return 'Random SMILES'

    @staticmethod
    def get_parameters():
        parameters = list()
        parameters.append({'id':'n', 'name':'Number of molecules', 'type':int})
        parameters.append({'id':'max_length', 'name':'Maximum length', 'type':int})
        return parameters

    @staticmethod
    def check_prerequisites(global_parameters, parameters):
        pass

    @staticmethod
    def execute():
        # TODO
        pass
