from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy, AUC
from utils import turn_gpu_off, reset_session
from utils.builders import CustomBuilder
from utils.data_loader import DatasetFlower
from shutil import move
from os.path import join, isfile
# because settings are ignored
settings = __import__("settings")


NAMES = ['FM', 'iFM', 'CLK', 'CSK']


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
    for it, layer in enumerate([layers_ftl, layers_ift, layers_clk, layers_csk]):
        reset_session.session_reset()
        # CSK & blood - OOM error
        if it == 3 and dset == 'blood' and 'gpu' in suff:
            continue
        _layer = layer.copy()
        _layer.extend(head)
        builder = CustomBuilder(filename=f'{NAMES[it]}_{dset}_{suff}', filepath='results',
                                layers=_layer,
                                **settings.BUILD[dset])
        builder.compile_model(**settings.COMPILE)
        settings.TRAIN['call_stop_kwargs']['baseline'] = settings.BASELINES[dset]
        if dset == 'nist':
            data_path = join('Y://', 'NIST', 'by_class', 'train')
        if dset == 'blood':
            data_path = join('Y://', 'FCNN', 'archive_blood_cells', 'dataset2-master', 'images', 'TRAIN')
        if dset == 'scenes':
            data_path = join('Y://', 'FCNN', 'archive_natural_scenes', 'seg_train')
        flower = DatasetFlower(path=data_path, out_shape=settings.BUILD[dset]['input_shape'],
                               batch=settings.TRAIN['batch'], split=settings.TRAIN['validation_split'])
        builder.train_model(generator=flower.generator, validation=flower.validation_generator,
                            steps=flower.length, val_steps=flower.validation_length,
                            **settings.TRAIN)
        if dset == 'nist':
            data_path = join('Y://', 'NIST', 'by_class', 'test')
        if dset == 'blood':
            data_path = join('Y://', 'FCNN', 'archive_blood_cells', 'dataset2-master', 'images', 'TEST')
        if dset == 'scenes':
            data_path = join('Y://', 'FCNN', 'archive_natural_scenes', 'seg_test')
        flower = DatasetFlower(path=data_path, out_shape=settings.BUILD[dset]['input_shape'], batch=4)
        builder.evaluate_model(generator=flower.generator, steps=flower.length)
        # prefer comparison between different models on the same dataset
        builder.save_model_info('CPU - local PC (IP: 180)', summary=True)
        filename = builder.filename + '_trained.hdf5'
        path_check_orig = join('results', 'checkpoints', filename)
        path_check_save = join('P://', 'Zaklad1.SPMIO', 'JakubZ', 'checkpoints', 'vs_cls', filename)
        if not isfile(path_check_orig):
            continue
        move(path_check_orig, path_check_save)
    return 0


if __name__ == '__main__':
    suffix = {False: 'gpu', True: 'cpu-1'}
    for off in [True]:
        if off:
            turn_gpu_off.turn_gpu_off()
        for tries in range(3):
            for dset in ['scenes', 'blood', 'nist']:
                # main(dset, suffix[off])
                main(dset, 'cpu-1')
        # balance out the number of nist tries
        main('nist', 'cpu-1')
