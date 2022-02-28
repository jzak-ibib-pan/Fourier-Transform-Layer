from os import listdir
from pandas import DataFrame as df
from numpy import mean, std, median
from ast import literal_eval


STANDARD_KEYS = ['ftl-activation',
                 'ftl-calculate_abs',
                 'input_shape',
                 'use_imaginary',
                 'ftl-kernel_initializer',
                 ]


def clean_eva_line(line):
    # analyze evaluation lines
    # -1 - \n sign
    splits = line.replace(' ', '').split('||')[:-1]
    splits[0] = splits[0].split('--')[1]
    return splits


def extract_from_line(line):
    _line = line.strip().replace(' ', '')
    if 'Dataset' in line:
        # [-1] - dataset, [-1] - dot at the end
        return _line.split(':')[-1][:-1]
    if any(key in line for key in STANDARD_KEYS):
        return _line.split('-')[-1]
    return None


def analyze(lines, monitor=[]):
    result = {'Evaluation': {}}
    for monit in monitor:
        result.update({monit: None})
    # find evaluation lines
    index = 0
    while 'Evaluation' not in lines[index] and index < len(lines):
        index += 1
    keys = clean_eva_line(lines[index + 1])
    values = clean_eva_line(lines[index + 2])
    for key, value in zip(keys, values):
        result['Evaluation'].update({key: float(value)})
    for monit in monitor:
        index = 0
        while monit not in lines[index] and index < len(lines):
            index += 1
        if index == len(lines):
            continue
        monitored = lines[index]
        result[monit] = extract_from_line(monitored)
    return result


def main_activ_abs(use_abs=False, calculate='mean'):
    chosen_metric = 'cat_acc'
    frame = {'None': {},
             'tanh': {},
             'selu': {},
             'relu': {},
             }
    # set zeros for mean calculation
    for key in frame.keys():
        for dset in ['cifar10', 'fmnist', 'mnist']:
            frame[key].update({dset: []})
    monitor = ['input_shape', 'ftl-activation', 'ftl-calculate_abs', 'Dataset']
    for tries in range(10):
        filepath = f'../experiments/hyperparameter_optimization/results/abs_and_activation/try_0{tries}/'
        for filename in listdir(filepath):
            if filename == 'checkpoints':
                continue
            with open(filepath + filename, 'r') as fil:
                lines = fil.readlines()
            result = analyze(lines, monitor)
            if literal_eval(result['ftl-calculate_abs']) == (not use_abs) or literal_eval(result['input_shape']) == (32, 32, 3):
                continue
            frame[result['ftl-activation']][result['Dataset']].append(result['Evaluation'][chosen_metric])
    result = frame.copy()
    for key in frame.keys():
        for dset in ['cifar10', 'fmnist', 'mnist']:
            if calculate == 'mean':
                result[key][dset] = mean(frame[key][dset])
                continue
            if calculate == 'std':
                result[key][dset] = std(frame[key][dset])
                continue
            if calculate == 'median':
                result[key][dset] = median(frame[key][dset])
                continue
    return df.from_dict(data=result)


def main(use_imag=False, calculate='mean'):
    chosen_metric = 'cat_acc'
    frame = {'ones': {},
             'he_normal': {},
             'glorot_uniform': {},
             }
    # set zeros for mean calculation
    for key in frame.keys():
        for dset in ['cifar10', 'fmnist', 'mnist']:
            frame[key].update({dset: []})
    monitor = ['input_shape', 'ftl-kernel_initializer', 'ftl-use_imaginary', 'Dataset']
    for tries in range(10):
        filepath = f'../experiments/hyperparameter_optimization/results/imag_and_initialization/try_0{tries}/'
        for filename in listdir(filepath):
            if filename == 'checkpoints':
                continue
            with open(filepath + filename, 'r') as fil:
                lines = fil.readlines()
            result = analyze(lines, monitor)
            if literal_eval(result['ftl-use_imaginary']) == (not use_imag) or literal_eval(result['input_shape']) == (32, 32, 3):
                continue
            frame[result['ftl-kernel_initializer']][result['Dataset']].append(result['Evaluation'][chosen_metric])
    result = frame.copy()
    for key in frame.keys():
        for dset in ['cifar10', 'fmnist', 'mnist']:
            if calculate == 'mean':
                result[key][dset] = mean(frame[key][dset])
                continue
            if calculate == 'std':
                result[key][dset] = std(frame[key][dset])
                continue
            if calculate == 'median':
                result[key][dset] = median(frame[key][dset])
                continue
    return df.from_dict(data=result)


def prepare_for_latex(frames):
    frame_mean = frames[0].to_numpy()
    frame_std = frames[1].to_numpy()
    frame_median = frames[2].to_numpy()
    frame_top = frames[1].head()
    names = list(frame_top.index)
    print('Mean')
    for line_id in range(frame_mean.shape[0]):
        txt = f'{names[line_id]} & '
        me = frame_mean[line_id]
        st = frame_std[line_id]
        for value_me, value_st in zip(me, st):
            txt += f'{value_me:.3f} $\pm$ {value_st:.3f} & '
        txt = txt[:-2]
        txt += '\ \\'.replace(' ', '')
        print(txt)
    print('Median')
    for line_id in range(frame_median.shape[0]):
        txt = f'{names[line_id]} & '
        for value_me in frame_mean[line_id]:
            txt += f'{value_me:.3f} & '
        txt = txt[:-2]
        txt += '\ \\'.replace(' ', '')
        print(txt)


if __name__ == '__main__':
    for use in [False, True]:
        print(use)
        frames = []
        for calc in ['mean', 'std', 'median']:
            frames.append(main_activ_abs(use, calculate=calc))
        prepare_for_latex(frames)

    for use in [False, True]:
        print(use)
        frames = []
        for calc in ['mean', 'std', 'median']:
            frames.append(main(use, calculate=calc))
        prepare_for_latex(frames)