from scipy.ndimage import filters
from util import hdf5_util, file_util, logger, progressbar
import h5py


def apply_gauss(path, data_set_name, sigma):
    temp_path = file_util.get_temporary_file_path('gauss')
    file_util.copy_file(path, temp_path)
    hdf5_util.set_property(temp_path, data_set_name + '_gauss_sigma', sigma)
    file_h5 = h5py.File(temp_path, 'r+')
    data_set = file_h5[data_set_name]
    slices = list()
    for length in data_set.shape[:-1]:
        slices.append(slice(0,length))
    logger.log('Applying gauss')
    with progressbar.ProgressBar(data_set.shape[-1]) as progress:
        for i in range(data_set.shape[-1]):
            index = tuple(slices + [i])
            data_set[index] = filters.gaussian_filter(data_set[index], sigma=sigma)
            progress.increment()
    file_h5.close()
    file_util.copy_file(temp_path, path)
