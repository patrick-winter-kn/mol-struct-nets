from util import misc


def generate_data(data, batch_size):
    chunks = misc.chunk_by_size(len(data), batch_size)
    while True:
        for chunk in chunks:
            batch = slice(chunk['start'], chunk['end'])
            input = data[batch]
            output = data.classes(batch)
            yield input, output
        data.shuffle()


def number_chunks(data, batch_size):
    return len(misc.chunk_by_size(len(data), batch_size))
