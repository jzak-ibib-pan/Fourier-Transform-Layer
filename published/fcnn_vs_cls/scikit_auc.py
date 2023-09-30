from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy, AUC
from utils import turn_gpu_off, reset_session
from utils.builders import CustomBuilder
from utils.data_loader import DatasetLoader
import os
# because settings are ignored
settings = __import__("settings")
import ipykernel


NAMES = ['CSK', 'CLK', 'FM', 'iFM']


def main(dset, suff, tries):
    settings.COMPILE['metrics'] = [CategoricalAccuracy(),
                                   TopKCategoricalAccuracy(k=5, name='top-5'),
                                   AUC(multi_label=True, name='mAUC', num_thresholds=1000),
                                   AUC(multi_label=False, name='uAUC', num_thresholds=1000),
                                   ]
    layers_csk = [{'conv2d': {'filters': settings.aCSK_FILTERS[dset],
                              'kernel_size': settings.aCSK_KERNELS[dset],
                              'activation': 'relu',
                              'use_bias': True,
                              'kernel_initializer': "he_normal",
                              'bias_initializer': "zeros"}, }]
    layers_clk = [{'conv2d': {'filters': settings.aCLK_FILTERS[dset],
                              'kernel_size': settings.aCLK_KERNELS[dset],
                              'activation': 'relu',
                              'use_bias': True,
                              'kernel_initializer': "he_normal",
                              'bias_initializer': "zeros"}, }]
    layers_ftl = [{'ftl': {'activation': 'relu',
                           'kernel_initializer': 'he_normal',
                           'train_imaginary': False,
                           'inverse': False,
                           'use_bias': False,
                           'bias_initializer': 'zeros',
                           'calculate_abs': False,
                           'normalize_to_image_shape': False}, }]
    layers_ift = [{'ftl': {'activation': 'relu',
                           'kernel_initializer': 'he_normal',
                           'train_imaginary': False,
                           'inverse': True,
                           'use_bias': False,
                           'bias_initializer': 'zeros',
                           'calculate_abs': False,
                           'normalize_to_image_shape': False}, }]
    head = [{'flatten': {}},
            {'dense': {}}]

    for it, layer in enumerate([layers_csk, layers_clk, layers_ftl, layers_ift]):
        _layer = layer.copy()
        _layer.extend(head)
        settings.TRAIN['call_stop_kwargs']['baseline'] = settings.BASELINES[dset]
        x_test, y_test = DatasetLoader(out_shape=settings.BUILD[dset]['input_shape'],
                                       dataset_name=dset).test_data
        checkpoints = os.listdir('results\checkpoints')
        stop = False
        idx_check = -1
        while not stop and idx_check < len(checkpoints):
            idx_check += 1
            if NAMES[it] not in checkpoints[idx_check]:
                continue
            if dset not in checkpoints[idx_check]:
                continue
            if dset == 'mnist' and 'fmnist' in checkpoints[idx_check]:
                continue
            if dset == 'cifar10' and 'cifar100' in checkpoints[idx_check]:
                continue
            if suff not in checkpoints[idx_check]:
                continue
            if f'{tries:04d}' not in checkpoints[idx_check]:
                continue
            stop = True
        path_weights = os.path.join('results', 'checkpoints', checkpoints[idx_check])
        print(path_weights)
        builder = CustomBuilder(filename=f'{NAMES[it]}_{dset}_{suff}', filepath='results/auc',
                                layers=_layer,
                                batch=settings.TRAIN['batch'],
                                **settings.BUILD[dset],)
        # TODO: remember to run evaluate (built-in) at least once, then wrapper once; in total twice
        #  - first run is usually gibberish; probably caused by weight initialization
        builder.model.load_weights(path_weights)
        builder.compile_model(**settings.COMPILE)
        builder.model.evaluate(x=x_test, y=y_test, batch_size=settings.TRAIN['batch'])
        builder.evaluate_model(x_data=x_test, y_data=y_test, auc=True)
        # prefer comparison between different models on the same dataset
        builder.save_model_info('AUC keras num_thresholds increased', summary=True)
        reset_session.session_reset()
    return 0


if __name__ == '__main__':
    suffix = {False: 'gpu', True: 'cpu'}
    off = False
    if off:
        turn_gpu_off.turn_gpu_off()
    for dataset in ['mnist', 'fmnist', 'cifar10', 'cifar100']:
        for tries in range(5):
            main(dataset, suffix[off], tries)
