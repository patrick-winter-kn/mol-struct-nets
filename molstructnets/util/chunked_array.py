from util import misc


class ChunkedArray():

    def __init__(self, data_set, chunks, as_bool):
        self.data_set = data_set
        self.chunks = chunks
        self.as_bool = as_bool
        self.current_chunk_number = -1
        self.current_chunk = None
        self.load_next_chunk()

    def load_next_chunk(self):
        self.current_chunk = None
        self.current_chunk_number += 1
        self.current_chunk = misc.copy_into_memory(self.data_set, self.as_bool, False,
                                                   self.chunks[self.current_chunk_number]['start'],
                                                   self.chunks[self.current_chunk_number]['end'])

    def reset(self):
        if len(self.chunks) > 1:
            self.current_chunk = None
            self.current_chunk_number = -1
            self.load_next_chunk()

    def __len__(self):
        return len(self.current_chunk)

    def __getitem__(self, item):
        return self.current_chunk[item]

    @property
    def shape(self):
        return self.current_chunk.shape

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

    def get_overall_size(self):
        return len(self.data_set)

    def close(self):
        self.current_chunk = None
        self.current_chunk_number = -1
