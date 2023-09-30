from utils import turn_gpu_off, reset_session
from utils.builders import CustomBuilder
from utils.data_loader import DatasetLoader
# because settings are ignored
settings = __import__("settings")
import ipykernel


NAMES = ['CLK']


def main(dset, suff):
    layers_clk = [{'conv2d': {'filters': settings.aCLK_FILTERS[dset],
                              'kernel_size': settings.aCLK_KERNELS[dset],
                              'activation': 'relu',
                              'use_bias': True,
                              'kernel_initializer': "he_normal",
                              'bias_initializer': "zeros"}, }]
    head = [{'flatten': {}},
            {'dense': {}}]

    for it, layer in enumerate([layers_clk]):
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
        builder.save_model_info('First article comparison', summary=True)
        reset_session.session_reset()
    return 0


if __name__ == '__main__':
    suffix = {False: 'gpu', True: 'cpu'}
    off = False
    if off:
        turn_gpu_off.turn_gpu_off()
    for dataset in ['cifar100']:
        for tries in range(20):
            main(dataset, suffix[off])
