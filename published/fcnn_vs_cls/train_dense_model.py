from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy, AUC
from utils import turn_gpu_off, reset_session
from utils.builders import CustomBuilder
from utils.data_loader import DatasetLoader, DatasetFlower
from os.path import join
# because settings are ignored
settings = __import__("settings")
import ipykernel


NAMES = ['dense']


def main(dset, suff):
    settings.COMPILE['metrics'] = [CategoricalAccuracy(),
                                   TopKCategoricalAccuracy(k=5, name='top-5'),
                                   AUC(multi_label=True, name='mAUC', num_thresholds=1000),
                                   AUC(multi_label=False, name='uAUC', num_thresholds=1000),
                                   ]
    head = [{'flatten': {}},
            {'dense': {}}]

    for it, layer in enumerate([[]]):
        _layer = layer.copy()
        _layer.extend(head)
        builder = CustomBuilder(filename=f'{NAMES[it]}_{dset}_{suff}', filepath='results',
                                layers=_layer,
                                **settings.BUILD[dset])
        builder.compile_model(**settings.COMPILE)
        settings.TRAIN['call_stop_kwargs']['baseline'] = settings.BASELINES[dset]
        if dset in ['mnist', 'fmnist', 'cifar10', 'cifar100']:
            x_train, y_train, x_test, y_test = DatasetLoader(out_shape=settings.BUILD[dset]['input_shape'],
                                                             dataset_name=dset).full_data
            builder.train_model(x_data=x_train, y_data=y_train,
                                **settings.TRAIN)
            builder.evaluate_model(x_data=x_test, y_data=y_test)
        else:
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
        builder.save_model_info('Dense model training comparison', summary=True)
        reset_session.session_reset()
    return 0


if __name__ == '__main__':
    suffix = {False: 'gpu', True: 'cpu'}
    off = False
    if off:
        turn_gpu_off.turn_gpu_off()
    for dataset in ['scenes', 'blood', 'mnist', 'fmnist', 'cifar10', 'cifar100', 'nist']:
        for tries in range(1):
            main(dataset, suffix[off])
