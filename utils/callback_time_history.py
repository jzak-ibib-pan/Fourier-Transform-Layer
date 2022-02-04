from tensorflow.keras.callbacks import Callback
from time import time


class TimeHistory(Callback):
    def __init__(self):
        super(TimeHistory, self).__init__()
        self.times = []
        self._epoch_time_start = 0

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self._epoch_time_start = time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time() - self._epoch_time_start)


if __name__ == '__main__':
    print(0)