from keras import callbacks

from util import file_util


class CustomCheckpoint(callbacks.Callback):

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs=None):
        save_model(self.model, self.file_path, epoch)


def save_model(model, path, epoch):
    temp_path = file_util.get_temporary_file_path('model')
    model.save(temp_path)
    file_util.move_file(temp_path, path)
    with open(path[:-3] + '-epochs.txt', 'w') as file:
        file.write(str(epoch + 1))
