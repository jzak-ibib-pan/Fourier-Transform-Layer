from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy, AUC
from utils import turn_gpu_off, reset_session
from utils.builders import CustomBuilder
from utils.data_loader import DatasetLoader
from shutil import move
from os.path import join, isfile
# because settings are ignored
settings = __import__("settings")


NAMES = ['CSK', 'CLK', 'FM', 'iFM']


def main(dset, suff):
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

    #for it, layer in enumerate([layers_csk, layers_clk, layers_ftl, layers_ift]):
    for it, layer in enumerate([layers_ftl, layers_ift]):
        reset_session.session_reset()
        _layer = layer.copy()
        _layer.extend(head)
        builder = CustomBuilder(filename=f'{NAMES[it]}_{dset}_{suff}', filepath='results',
                                layers=_layer,
                                **settings.BUILD[dset])
        builder.compile_model(**settings.COMPILE)
        settings.TRAIN['call_stop_kwargs']['baseline'] = settings.BASELINES[dset]
        x_train, y_train, x_test, y_test = DatasetLoader(out_shape=settings.BUILD[dset]['input_shape'],
                                                         dataset_name=dset).full_data
        builder.train_model(x_data=x_train, y_data=y_train,
                            **settings.TRAIN)
        builder.evaluate_model(x_data=x_test, y_data=y_test)
        # prefer comparison between different models on the same dataset
        builder.save_model_info('CPU - local PC (IP: 180)', summary=True)
        filename = builder.filename + '_trained.hdf5'
        path_check_orig = join('results', 'checkpoints', filename)
        path_check_save = join('P://', 'Zaklad1.SPMIO' ,'JakubZ', 'checkpoints', 'vs_cls', filename)
        if not isfile(path_check_orig):
            continue
        move(path_check_orig, path_check_save)
    return 0


if __name__ == '__main__':
    suffix = {False: 'gpu', True: 'cpu-1'}
    for off in [False]:
        if off:
            turn_gpu_off.turn_gpu_off()
        #for dataset in ['mnist', 'fmnist', 'cifar10', 'cifar100']:
        for dataset in ['mnist']:
            for tries in range(10):
                main(dataset, suffix[off])
