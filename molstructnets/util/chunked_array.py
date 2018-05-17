from util import misc, logger


class ChunkedArray():

    def __init__(self, data_set, chunks, as_bool):
        self.data_set = data_set
        self.chunks = chunks
        self.as_bool = as_bool
        self.current_chunk_number = -1
        self.current_chunk = None

    def load_next_chunk(self):
        self.current_chunk = None
        self.current_chunk_number += 1
        self.current_chunk = misc.copy_into_memory(self.data_set, self.as_bool, False,
                                                   self.chunks[self.current_chunk_number]['start'],
                                                   self.chunks[self.current_chunk_number]['end'],
                                                   log_level=logger.LogLevel.DEBUG)

    def reset(self):
        self.current_chunk = None
        self.current_chunk_number = -1

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

    def has_next(self):
        return self.current_chunk_number + 1 < len(self.chunks)

    def min(self, *args, **kwargs):
        return self.current_chunk.min(*args, **kwargs)

    def max(self, *args, **kwargs):
        return self.current_chunk.max(*args, **kwargs)

    def mean(self, *args, **kwargs):
        return self.current_chunk.mean(*args, **kwargs)

    def unload(self):
        self.current_chunk = None

    def copy_chunk(self):
        return self.current_chunk.copy()
