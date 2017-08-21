class Repository:

    def __init__(self):
        self._implementations = []
        self._implementation_ids = []

    def add_implementation(self, implementation):
        self._implementations.append(implementation)
        self._implementation_ids.append(implementation.get_id())

    def get_implementations(self):
        return self._implementations

    def get_implementation(self, id_):
        return self._implementations[self._implementation_ids.index(id_)]
