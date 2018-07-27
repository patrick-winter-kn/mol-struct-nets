from util import misc, logger


class ChunkedArray:

    def __init__(self, data_set, chunks, as_bool):
        self.data_set = data_set
        self.chunks = chunks
        self.as_bool = as_bool
        self.current_chunk_number = -1
        self.current_chunk = None

    def load_chunk(self, chunk_number):
        if self.current_chunk_number != chunk_number or self.current_chunk is None:
            self.current_chunk = None
            self.current_chunk_number = chunk_number
            self.current_chunk = misc.copy_into_memory(self.data_set, self.as_bool, False,
                                                       self.chunks[self.current_chunk_number]['start'],
                                                       self.chunks[self.current_chunk_number]['end'],
                                                       log_level=logger.LogLevel.DEBUG)
            return True
        else:
            return False

    def write_current_chunk(self):
        current_chunk = self.chunks[self.current_chunk_number]
        self.data_set[current_chunk['start']:current_chunk['end']] = self.current_chunk[:]

    def __len__(self):
        return len(self.current_chunk)

    def __getitem__(self, item):
        return self.current_chunk[item]

    @property
    def shape(self):
        return self.current_chunk.shape

    @property
    def original_shape(self):
        return self.data_set.shape

    def number_chunks(self):
        return len(self.chunks)

    def get_current_chunk_number(self):
        return self.current_chunk_number

    def get_chunks(self):
        return self.chunks

    def unload(self):
        self.current_chunk = None
