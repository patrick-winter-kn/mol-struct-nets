import multiprocessing


class BufferedQueue():

    def __init__(self, buffer_size):
        self._queue = multiprocessing.Manager().Queue()
        self._values = list()
        self._buffer_size = buffer_size
        self._received = list()

    def put(self, value):
        self._values.append(value)
        if len(self._values) >= self._buffer_size:
            self._queue.put(self._values)
            self._values = list()

    def get(self):
        if self._received is None or len(self._received) == 0:
            self._received = self._queue.get()
        if self._received is None or len(self._received) == 0:
            return None
        else:
            return self._received.pop(0)

    def flush(self):
        if len(self._values) > 0:
            self._queue.put(self._values)
            self._values = list()
