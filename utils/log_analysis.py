from os import listdir
from pandas import DataFrame as df
from numpy import mean, std, median


STANDARD_KEYS = ['ftl-activation',
                 'ftl-calculate_abs',
                 'input_shape',
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


def main(use_abs=False, calculate='mean'):
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
        filepath = f'../experiments/hyperparameter_optimization/results/try_0{tries}/'
        for filename in listdir(filepath):
            if filename == 'checkpoints':
                continue
            with open(filepath + filename, 'r') as fil:
                lines = fil.readlines()
            result = analyze(lines, monitor)
            # remember to remove spaces - '' work, str - does not
            if result['ftl-calculate_abs'] == str(not use_abs) or result['input_shape'] == '(32,32,3)':
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


def prepare_for_latex(frames):
    frame_mean = frames[0].to_numpy()
    frame_std = frames[1].to_numpy()
    frame_median = frames[2].to_numpy()
    print('Mean')
    for line_id in range(frame_mean.shape[0]):
        txt = ''
        me = frame_mean[line_id]
        st = frame_std[line_id]
        for value_me, value_st in zip(me, st):
            txt += f'{value_me:.3f} $\pm$ {value_st:.3f} & '
        txt = txt[:-2]
        txt += '\ \\'.replace(' ', '')
        print(txt)
    print('\n')
    print('Median')
    for line_id in range(frame_median.shape[0]):
        txt = ''
        for value_me in frame_mean[line_id]:
            txt += f'{value_me:.3f} & '
        txt = txt[:-2]
        txt += '\ \\'.replace(' ', '')
        print(txt)


if __name__ == '__main__':
    frames = []
    for calc in ['mean', 'std', 'median']:
        frames.append(main(calculate=calc))
    prepare_for_latex(frames)
