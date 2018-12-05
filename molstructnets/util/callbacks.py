from keras import callbacks
import time

from util import file_util


class CustomCheckpoint(callbacks.Callback):

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        end_time = time.time()
        time_difference = end_time - self.start_time
        save_model(self.model, self.file_path, epoch, time_difference)


def save_model(model, path, epoch, epoch_time=0):
    temp_path = file_util.get_temporary_file_path('model')
    model.save(temp_path)
    file_util.move_file(temp_path, path)
    epochs_file_path = path[:-3] + '-epochs.txt'
    previous_time = 0
    if file_util.file_exists(epochs_file_path):
        with open(epochs_file_path, 'r') as file:
            lines = file.readlines()
            if len(lines) > 1:
                previous_time = float(lines[1])
    with open(epochs_file_path, 'w') as file:
        file.write(str(epoch + 1) + '\n' + str(previous_time + epoch_time))
