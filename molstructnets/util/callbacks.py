from keras import callbacks

from util import file_util


class CustomCheckpoint(callbacks.Callback):

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs=None):
        temp_path = file_util.get_temporary_file_path('model')
        self.model.save(temp_path)
        file_util.move_file(temp_path, self.file_path)
        with open(self.file_path[:-3] + '-epochs.txt', 'w') as file:
            file.write(str(epoch + 1))
