# due to error, all files before 12.04.2022 were calculated on CPU. This script renames the files
import os


def main(path):
    loof_files = os.listdir(path)
    for filename in loof_files:
        if 'cpu' not in filename:
            continue
        target = filename.replace('cpu', 'cpu-1')
        path_file = os.path.join(path, filename)
        path_target = os.path.join(path, target)
        os.rename(path_file, path_target)


if __name__ == '__main__':
    # main(os.path.join('P://', 'Zaklad1.SPMIO', 'JakubZ', 'checkpoints', 'vs_cls'))
    # main(os.path.join('H://', 'Spyder', 'fcnn', 'experiments', 'fcnn_vs_cls', 'results'))
    # main(os.path.join('H://', 'Spyder', 'fcnn', 'experiments', 'fcnn_vs_cls', 'results', 'auc'))
    # main(os.path.join('H://', 'Spyder', 'fcnn', 'experiments', 'fcnn_vs_cls', 'results', 'auc', 'old'))
    # main(os.path.join('H://', 'Spyder', 'fcnn', 'experiments', 'fcnn_vs_cls', 'results', 'auc', 'old', 'default_num_threshold_200'))
    # main(os.path.join('H://', 'Spyder', 'fcnn', 'experiments', 'fcnn_vs_cls', 'results', 'auc', 'old','num_thresh_1000'))
    # main(os.path.join('H://', 'Spyder', 'fcnn', 'experiments', 'fcnn_vs_cls', 'results', 'auc', 'old','num_thresh_500'))
    main(os.path.join('H://', 'Spyder', 'fcnn', 'experiments', 'fcnn_vs_cls', 'results', 'less_params_than_ftl'))