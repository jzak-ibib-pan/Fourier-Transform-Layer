def analyze(lines, monitor=[]):
    for line in lines:
        for monit in monitor:
            if monit not in line:
                continue
            print(line)
    eva_index = 0
    while 'Evaluation' not in lines[eva_index] and eva_index < len(lines):
        eva_index += 1
    print(lines[eva_index + 1 : eva_index + 3])


def main():
    monitor = ['ftl-activation', 'Dataset']
    filename = 'mnist_abs-calc_True_2022-02-23_15_37_52_007.txt'
    filepath = '../experiments/hyperparameter_optimization/results/try_00/'
    with open(filepath + filename, 'r') as fil:
        lines = fil.readlines()
    analyze(lines, monitor)


if __name__ == '__main__':
    main()