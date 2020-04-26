import os
import sys
import re
import random
import itertools
import subprocess
import math
import numpy as np
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from train import get_args
from util import util


def args2list(args):
    return [
        "--data-path", str(args.data_path),
        '--task', str(args.task),
        '--language', str(args.language),
        '--batch-size', str(args.batch_size),
        "--representation", str(args.representation),
        '--eval-batches', str(args.eval_batches),
        '--wait-epochs', str(args.wait_epochs),
        "--checkpoint-path", str(args.checkpoint_path),
        "--seed", str(args.seed),
    ]


def get_hyperparameters(search):
    hyperparameters = {
        '--hidden-size': search[0],
        '--nlayers': search[1],
        '--dropout': search[2],
        '--pca-size': search[3],
    }
    return dict2list(hyperparameters)


def get_hyperparameters_search(n_runs, representation):
    bert_pca_size = list([768])
    fast_pca_size = list([300])
    onehot_pca_size = list({int(2**x) for x in np.arange(5.6, 8.2, 0.01)})
    hidden_size = list({int(2**x) for x in np.arange(2, 9, 0.01)})
    nlayers = [1, 2, 3]
    dropout = list(np.arange(0.0, 0.51, 0.01))

    if representation in ['onehot', 'random']:
        pca_size = onehot_pca_size
    elif representation == 'fast':
        pca_size = fast_pca_size
    elif representation == 'bert':
        pca_size = bert_pca_size
    else:
        raise ValueError('Invalid representation %s' % representation)

    all_hyper = [hidden_size, nlayers, dropout, pca_size]
    grid = list(itertools.product(*all_hyper))

    return random.sample(grid, n_runs)


def dict2list(data):
    list2d = [[k, str(x)] for k, x in data.items()]
    return list(itertools.chain.from_iterable(list2d))


def write_done(done_fname):
    with open(done_fname, "w") as f:
        f.write('done training\n')


def append_result(fname, values):
    with open(fname, "a+") as f:
        f.write(','.join(values) + '\n')


def get_results(out, err):
    loss_pattern = '^Final loss. Train: (\d.\d+) Dev: (\d.\d+) Test: (\d.\d+)$'
    acc_pattern = '^Final acc. Train: (\d.\d+) Dev: (\d.\d+) Test: (\d.\d+)$'

    output = out.decode().split('\n')

    try:
        m = re.match(loss_pattern, output[-3])
        train_loss, dev_loss, test_loss = m.groups()

        m = re.match(acc_pattern, output[-2])
        train_acc, dev_acc, test_acc = m.groups()
    except:
        print('Output:', output)
        raise ValueError('Error in subprocess: %s' % err.decode())

    return [train_loss, dev_loss, test_loss, train_acc, dev_acc, test_acc]


def main():
    args = get_args()
    n_runs = 50

    ouput_path = os.path.join(args.checkpoint_path, args.task, args.language, args.representation)
    results_fname = os.path.join(ouput_path, 'all_results.txt')
    done_fname = os.path.join(ouput_path, 'finished.txt')

    curr_iter = util.file_len(results_fname) - 1
    util.mkdir(ouput_path)

    if curr_iter == -1:
        res_columns = ['hidden_size', 'nlayers', 'dropout', 'pca_size',
                       'train_loss', 'dev_loss', 'test_loss',
                       'train_acc', 'dev_acc', 'test_acc']
        append_result(results_fname, res_columns)
        curr_iter = 0

    search = get_hyperparameters_search(n_runs, args.representation)

    for hyper in tqdm(search[curr_iter:], initial=curr_iter, total=n_runs):
        hyperparameters = get_hyperparameters(hyper)

        my_env = os.environ.copy()
        cmd = ['python', 'src/h02_learn/train.py'] + args2list(args) + hyperparameters
        tqdm.write(str(hyperparameters))
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env)
        out, err = process.communicate()

        results = get_results(out, err)
        append_result(results_fname, [str(x) for x in hyper] + results)

    write_done(done_fname)


if __name__ == '__main__':
    main()
