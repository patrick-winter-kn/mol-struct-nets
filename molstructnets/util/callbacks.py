from keras import callbacks
import gc


class CustomCheckpoint(callbacks.Callback):

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        self.model.save(self.file_path, overwrite=True)
        with open(self.file_path[:-3] + '-epochs.txt', 'w') as file:
            file.write(str(epoch + 1))
