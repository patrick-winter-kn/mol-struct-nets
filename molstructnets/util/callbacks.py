from keras import callbacks


class CustomCheckpoint(callbacks.Callback):

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.file_path, overwrite=True)
        with open(self.file_path[:-3] + '-epochs.txt') as file:
            file.write(str(epoch + 1))
