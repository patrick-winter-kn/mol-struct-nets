from scipy.ndimage import filters
from util import hdf5_util, file_util, logger, progressbar, misc
import h5py


def apply_gauss(path, data_set_name, sigma):
    temp_path = file_util.get_temporary_file_path('gauss')
    file_util.copy_file(path, temp_path)
    hdf5_util.set_property(temp_path, data_set_name + '_gauss_sigma', sigma)
    file_h5 = h5py.File(temp_path, 'r+')
    data_set = file_h5[data_set_name]
    chunked_data = misc.get_chunked_array(data_set, fraction=0.5)
    logger.log('Applying gauss (in ' + str(chunked_data.number_chunks()) + ' chunks)')
    with progressbar.ProgressBar(chunked_data.number_chunks() * 2 + chunked_data.number_chunks()
                                 * data_set.shape[-1]) as progress:
        for i in range(chunked_data.number_chunks()):
            chunked_data.load_chunk(i)
            progress.increment()
            data = chunked_data[:]
            slices = list()
            for length in data.shape[:-1]:
                slices.append(slice(0,length))
            for j in range(data.shape[-1]):
                index = tuple(slices + [j])
                data[index] = filters.gaussian_filter(data[index], sigma=sigma)
                progress.increment()
            chunked_data.write_current_chunk()
            progress.increment()
    chunked_data.unload()
    file_h5.close()
    file_util.copy_file(temp_path, path)
