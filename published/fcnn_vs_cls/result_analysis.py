from os.path import join
from os import listdir
import numpy as np
from tensorflow.keras.metrics import AUC, CategoricalAccuracy, TopKCategoricalAccuracy
from utils import turn_gpu_off, reset_session
from utils.builders import CustomBuilder
from utils.data_loader import DatasetLoader
# because settings are ignored
settings = __import__("settings")


NAMES = ['CSK', 'CLK', 'FM', 'iFM']


def main_auc(dset, suff):
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
    x_test, y_test = DatasetLoader(out_shape=settings.BUILD[dset]['input_shape'],
                                   dataset_name=dset).test_data
    for it, layer in enumerate([layers_csk, layers_clk, layers_ftl, layers_ift]):
        # path_w = 'results/checkpoints'
        path_w = join('P://', 'Zaklad1.SPMIO', 'JakubZ', 'checkpoints', 'vs_cls')
        name = NAMES[it]
        # caused errors - can find both cifar10 and cifar100 and FM and iFM
        weights = [l for l in listdir(path_w) if f'{name}_{dset}' in l]
        if dset == 'mnist':
            weights = [l for l in weights if 'fmnist' not in l]
        if dset == 'cifar10':
            weights = [l for l in weights if 'cifar100' not in l]
        if name == 'FM':
            weights = [l for l in weights if 'iFM' not in l]
        # iterate over saved weights
        _layer = layer.copy()
        _layer.extend(head)
        for weight in weights:
            if 'cpu' in weight:
                continue
            reset_session.session_reset()
            # https://github.com/keras-team/keras/issues/6977
            # TODO: remember to run evaluate (built-in) at least once, then wrapper once; in total twice
            #  - first run is usually gibberish; probably caused by weight initialization
            builder = CustomBuilder(filename=suff + weight[:-13], filepath='results/auc',
                                    layers=_layer,
                                    **settings.BUILD[dset])
            builder.compile_model(**settings.COMPILE)
            builder.model.load_weights(join(path_w, weight))
            builder.model.evaluate(x=x_test, y=y_test, batch_size=settings.TRAIN['batch'])
            builder.evaluate_model(x_data=x_test, y_data=y_test)
            # prefer comparison between different models on the same dataset
            builder.save_model_info('AUC evaluation test', summary=True)
    return 0


def manual_analysis(model_type='FM', dset='mnist'):
    layers = {'iFM': [{'ftl': {'activation': 'relu',
                               'kernel_initializer': 'he_normal',
                               'train_imaginary': False,
                               'inverse': True,
                               'use_bias': False,
                               'bias_initializer': 'zeros',
                               'calculate_abs': False,
                               'normalize_to_image_shape': False}
                      }],
              'FM': [{'ftl': {'activation': 'relu',
                              'kernel_initializer': 'he_normal',
                              'train_imaginary': False,
                              'inverse': False,
                              'use_bias': False,
                              'bias_initializer': 'zeros',
                              'calculate_abs': False,
                              'normalize_to_image_shape': False}
                     }],
              'CSK': [{'conv2d': {'filters': settings.aCSK_FILTERS[dset],
                                  'kernel_size': settings.aCSK_KERNELS[dset],
                                  'activation': 'relu',
                                  'use_bias': True,
                                  'kernel_initializer': "he_normal",
                                  'bias_initializer': "zeros"},
                       }],
              'CLK': [{'conv2d': {'filters': settings.aCLK_FILTERS[dset],
                                  'kernel_size': settings.aCLK_KERNELS[dset],
                                  'activation': 'relu',
                                  'use_bias': True,
                                  'kernel_initializer': "he_normal",
                                  'bias_initializer': "zeros"},
                       }],
              }

    head = [{'flatten': {}},
            {'dense': {}}]
    _layers = layers[model_type].copy()
    _layers.extend(head)
    # caused errors - can find both cifar10 and cifar100 and FM and iFM
    # iterate over saved weights
    x_train, y_train, x_test, y_test = DatasetLoader(out_shape=settings.BUILD[dset]['input_shape'],
                                                     dataset_name=dset).full_data
    # https://github.com/keras-team/keras/issues/6977
    wpath = 'results/auc'
    weights = [li for li in listdir('results/checkpoints') if f'{model_type}_{dset}' in li]
    if dset == 'mnist':
        weights = [li for li in weights if 'fmnist' not in li]
    if dset == 'cifar10':
        weights = [li for li in weights if 'cifar100' not in li]
    if model_type == 'FM':
        weights = [li for li in weights if 'iFM' not in li]
    for filename in weights:
        reset_session.session_reset()
        builder = CustomBuilder(filename='auc_' + filename[:-13], filepath=wpath,
                                layers=_layers,
                                input_shape=settings.BUILD[dset]['input_shape'],
                                noof_classes=settings.BUILD[dset]['noof_classes'])
        builder.compile_model(**{'optimizer': 'adam',
                                 'loss': 'categorical_crossentropy',
                                 'metrics': [CategoricalAccuracy(),
                                             AUC(multi_label=True, name='mAUC'),
                                             AUC(multi_label=False, name='uAUC')
                                             ],
                                 'run_eagerly': False,
               })
        from os.path import join
        builder.model.load_weights(join('results', 'checkpoints', filename))
        builder.evaluate_model(x_data=x_test, y_data=y_test)
        print('After loading')
        print(builder.evaluation)
        builder.save_model_info(summary=True)


def main_history(path='results', models=('CSK', 'CLK', 'FM', 'iFM'), dsets=('mnist', 'fmnist', 'cifar10', 'cifar100'),
                 auc=False, processing='gpu'):
    loof_files = sorted([li for li in listdir(path) if 'auc' not in li])
    results = {}
    for key in models:
        if key not in results.keys():
            results.update({key: {}})
        for dset in dsets:
            results[key].update({dset : {'eval': [], 'mAUC': [], 'uAUC': [], 'mAUCsci': [], 'wAUCsci': [], 'times': [], 'epochs': []}})
    for filename in loof_files:
        for dset in dsets:
            if processing not in filename:
                continue
            if dset not in filename:
                continue
            if dset == 'nist' and ('fmnist' in filename or 'mnist' in filename):
                continue
            if dset == 'mnist' and 'fmnist' in filename:
                continue
            if dset == 'cifar10' and 'cifar100' in filename:
                continue
            for model in results.keys():
                if model not in filename:
                    continue
                if model == 'FM' and 'iFM' in filename:
                    continue
                print(filename)
                with open(join(path, filename), 'r') as fil:
                    data = fil.readlines()
                idx_eval = 0
                while 'Evaluation' not in data[idx_eval]:
                    idx_eval += 1
                # +2 - two additional lines
                eval_splits = data[idx_eval + 2].split('||')
                results[model][dset]['eval'].append(float((eval_splits[1]).strip()))
                if 'auc' in path or auc:
                    idx_auc = 3
                    if 'auc' in path:
                        idx_auc = 2
                    results[model][dset]['mAUC'].append(float((eval_splits[idx_auc]).strip()))
                    results[model][dset]['uAUC'].append(float((eval_splits[idx_auc + 1]).strip()))
                    if len(eval_splits) > 6:
                        results[model][dset]['mAUCsci'].append(float((eval_splits[5]).strip()))
                        results[model][dset]['wAUCsci'].append(float((eval_splits[6]).strip()))
                idx_times = idx_eval + 5
                noof_epochs = 0
                while 'Epoch' in data[idx_times]:
                    time_splits = data[idx_times].split('||')
                    results[model][dset]['times'].append(float((time_splits[-2]).strip()))
                    idx_times += 1
                    noof_epochs += 1
                results[model][dset]['epochs'].append(noof_epochs)
    rounds = {'times': 1,
              'epochs': 0,
              'default': 2,
              }
    for key in results.keys():
        print(key)
        txt = {}
        for dset in dsets:
            for key_interior in ['eval', 'mAUC', 'uAUC', 'mAUCsci', 'wAUCsci', 'times', 'epochs']:
                if key_interior not in txt.keys():
                    txt.update({key_interior: ''})
                if len(results[key][dset][key_interior]) < 1:
                    continue
                _round = [rounds['default'] if key_interior not in rounds.keys() else rounds[key_interior]][0]
                _mult = [1 if key_interior in ['times', 'epochs'] else 100][0]
                txt[key_interior] += f'${(_mult * np.mean(results[key][dset][key_interior])):.{_round}f} \pm '
                txt[key_interior] += '{'
                txt[key_interior] += f'{(_mult * np.std(results[key][dset][key_interior])):.{_round}f}'
                txt[key_interior] += '}$ & '
            extras = []
            start = 0
            # zip caused incorrect values
            for ep in results[key][dset]['epochs']:
                runtimes = results[key][dset]['times'][start: start + ep]
                extras.append(np.mean(runtimes) * ep)
                start = ep
            if 'extras' not in txt.keys():
                txt.update({'extras': ''})
            if len(extras) < 1:
                continue
            txt['extras'] += f'${np.mean(extras):.1f} \pm'
            txt['extras'] += '{'
            txt['extras'] += f'{np.std(extras):.1f}'
            txt['extras'] += '}$ & '
        for key_interior in txt.keys():
            if txt[key_interior] == '':
                continue
            # print(f'{key_interior}\n{txt[key_interior]}')
        # print('#####\n')
    # dataset table
    for dset in dsets:
        print(dset)
        txt = {}
        for key in results.keys():
            for key_interior in ['eval', 'mAUC', 'uAUC', 'mAUCsci', 'wAUCsci', 'times', 'epochs']:
                if key_interior not in txt.keys():
                    txt.update({key_interior: ''})
                if len(results[key][dset][key_interior]) < 1:
                    continue
                _round = [rounds['default'] if key_interior not in rounds.keys() else rounds[key_interior]][0]
                _mult = [1 if key_interior in ['times', 'epochs'] else 100][0]
                txt[key_interior] += f'${(_mult * np.mean(results[key][dset][key_interior])):.{_round}f} \pm '
                txt[key_interior] += '{'
                txt[key_interior] += f'{(_mult * np.std(results[key][dset][key_interior])):.{_round}f}'
                txt[key_interior] += '}$ & '
            extras = []
            start = 0
            # zip caused incorrect values
            for ep in results[key][dset]['epochs']:
                runtimes = results[key][dset]['times'][start: start + ep]
                extras.append(np.mean(runtimes) * ep)
                start = ep
            if 'extras' not in txt.keys():
                txt.update({'extras': ''})
            if len(extras) < 1:
                continue
            txt['extras'] += f'${np.mean(extras):.1f} \pm'
            txt['extras'] += '{'
            txt['extras'] += f'{np.std(extras):.1f}'
            txt['extras'] += '}$ & '
        for key_interior in txt.keys():
            if txt[key_interior] == '':
                continue
            print(f'{key_interior}\n{txt[key_interior]}')
        print('#####\n')


if __name__ == '__main__':
    # settings.COMPILE['metrics'].append(AUC(multi_label=True, name='mAUC'))
    # settings.COMPILE['metrics'].append(AUC(multi_label=False, name='uAUC'))
    # for dataset in ['cifar10', 'mnist', 'fmnist']:
    #     main_auc(dataset, 'auc_')
    # ['nist', 'scenes', 'blood']
    # ['mnist', 'fmnist', 'cifar10', 'cifar100']
    # for model in ['CSK', 'CLK', 'FM', 'iFM']:
    #     for dataset in ['cifar100', 'mnist', 'fmnist', 'cifar10']:
    #         manual_analysis(model, dataset)
    # ['nist', 'scenes', 'blood']
    main_history(path='results',
                 models=['CSK', 'CLK', 'FM', 'iFM'],
                 dsets=['nist', 'scenes', 'blood'],
                 # dsets=['nist', 'scenes', 'blood'],
                 auc=True,
                 processing='cpu-1')
    # only required for cpu, gpu has built in AUC calculation - but for typical dsets num_thresh was incorrectly setup,
    # thus gpu also had to be recalculated
    # for dataset in ['cifar100', 'cifar10', 'mnist', 'fmnist']:
    #     main_auc(dataset, 'auc_')
