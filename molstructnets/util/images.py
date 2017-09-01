import numpy
from util import progressbar, file_util
from keras.preprocessing import image


def load_images(image_directory, width, height, start, end, show_progress_bar=False):
    n = end - start
    img_array = numpy.zeros((n, width, height, 3), dtype=numpy.uint8)
    if show_progress_bar:
        progress = progressbar.ProgressBar(n)
    else:
        progress = None
    index = 0
    for i in range(start, end):
        img = image.load_img(file_util.resolve_subpath(image_directory, str(i) + '.png'))
        img_array[index] = image.img_to_array(img)
        index += 1
        if progress is not None:
            progress.increment()
    if progress is not None:
        progress.finish()
    return img_array
