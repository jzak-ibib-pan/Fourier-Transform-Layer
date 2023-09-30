from os.path import join, isfile
from os import listdir
import numpy as np


def extract(path='results', models=('CSK', 'CLK', 'FM', 'iFM'),
            dsets=('mnist', 'fmnist', 'cifar10', 'cifar100', 'nist', 'scenes', 'blood'),
            auc=True, processing='gpu'):
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

    fname = 'auc_for_AP.txt'
    opening = {False: 'w',
               True: 'a',
               }
    with open(fname, opening[isfile(fname)]) as file:
        file.write(f'{processing.upper()}\n')
        for key in results.keys():
            for dset in dsets:
                file.write(f'{results[key][dset]["mAUC"]}\n')

if __name__ == '__main__':
    extract(processing='gpu')
    extract(processing='cpu-1')