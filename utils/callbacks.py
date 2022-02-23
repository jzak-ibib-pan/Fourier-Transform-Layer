from tensorflow.keras.callbacks import Callback
from time import time
from numpy import isnan


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


class EarlyStopOnBaseline(Callback):
    """Callback that terminates training when either acc, loss, val_acc or val_loss reach a specified baseline
    (str) monitor - what to monitor: available for now: acc, loss, val_acc, val_loss; default: val_loss
    (float) baseline - expected value to reach before monitoring progress by delta; default: 0.1, range: 0 - 1
    (float) delta - how much the monitored value has to decrease/increase (loss/acc); default: 0.01, range: >= 0
    (int) patience - how many epochs with no better results to wait before stopping; default: 0, range: >= 0
    (bool) restore_best - whether to store and restore best weights; default: True
    (int) verbose - whether to print out the messages during training; default: 1, values: 0, 1
    """
    def __init__(self, monitor: str = 'val_loss', baseline: float = 0.1, min_delta: float = 0.001, patience: int = 0,
                 restore_best: bool = True, verbose: int = 1):
        super(EarlyStopOnBaseline, self).__init__()
        assert monitor in ['acc', 'loss', 'val_acc', 'val_loss', 'categorical_accuracy', 'val_categorical_accuracy',
                           'top_k_categorical_accuracy'], \
            self._inform_user_of_error('monitor')
        assert 0 < baseline < 1, self._inform_user_of_error('baseline')
        assert min_delta >= 0, self._inform_user_of_error('min_delta')
        assert patience >= 0, self._inform_user_of_error('patience')
        assert verbose in [0, 1], self._inform_user_of_error('verbose')
        self._monitor = monitor
        self._baseline = baseline
        self._delta = min_delta
        self._patience = patience
        # default - best weights will be restored
        self._restore_weights = restore_best
        self._verbose = verbose
        self._patience_counter = 0
        self._flag_monitor_accuracy = any([a in monitor for a in ['acc', 'accuracy']])
        # assume loss is the monitored value
        self._best_value = 1e10
        if self._flag_monitor_accuracy:
            self._best_value = 1e-10
        self._flag_reached_baseline = False
        self._best_weights = None
        self._stopped_training = False
        self._delta_sign = {True: 1,
                            False: -1,
                            }


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if isnan(logs.get('loss')):
            if self._verbose:
                print(f'\tEpoch {epoch}: Terminating training due to nan loss.')
            self.model.stop_training = True
            self._stopped_training = True

        monitored_value = logs.get(self._monitor)

        if monitored_value is None or self._baseline is None:
            return

        # find out if the monitored value improved
        if self._flag_monitor_accuracy:
            self._flag_reached_baseline = monitored_value >= self._baseline or self._flag_reached_baseline
            flag_better_result = monitored_value >= self._best_value + \
                                 self._delta * self._delta_sign[self._flag_monitor_accuracy]
        else:
            self._flag_reached_baseline = monitored_value <= self._baseline or self._flag_reached_baseline
            flag_better_result = monitored_value <= self._best_value + \
                                 self._delta * self._delta_sign[self._flag_monitor_accuracy]

        if not self._flag_reached_baseline:
            return

        # update values if reached new best
        if flag_better_result:
            self._best_value = monitored_value
            self._patience_counter = 0
            if self._restore_weights:
                self._best_weights = self.model.get_weights()
            return

        # restore best weights
        if self._restore_weights:
            self.model.set_weights(self._best_weights)
            if self._verbose:
                print(f'\tRestoring weights @ {self._monitor} = '
                      f'{round(self._best_value + self._delta * self._delta_sign[self._flag_monitor_accuracy], 4)} vs '
                      f'{round(monitored_value, 4)}.')
        self._patience_counter += 1
        if self._patience_counter < self._patience:
            return

        if self._verbose:
            print(f'\tEpoch {epoch}: Terminating training.')
        self.model.stop_training = True
        self._stopped_training = True

    def get_kwargs(self):
        return {'monitor': self._monitor,
                'baseline': self._baseline,
                'min_delta': self._delta,
                'patience': self._patience,
                'restore_best': self._restore_weights,
                }

    @staticmethod
    def _inform_user_of_error(variable: str):
        RANGES = {'monitor': ['acc', 'loss', 'val_acc', 'val_loss', 'categorical_accuracy', 'val_categorical_accuracy',
                              'top_k_categorical_accuracy'],
                  'baseline': '(0;1)',
                  'min_delta': '[0;+inf)',
                  'patience': '[0;+inf)',
                  'verbose': '0, 1',
                  }
        return f'Incorrect value. Expected value in range: {RANGES[variable]}.'

    @property
    def stopped_training(self):
        return self._stopped_training


if __name__ == '__main__':
    print(0)