import os
import sys
import argparse
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h02_learn.dataset import get_data_loaders
from h02_learn.model import MLP, TransparentDataParallel
from h02_learn.train_info import TrainInfo
from util import util
from util import constants


def get_model_name(args):
    fpath = 'nl_%d-es_%d-hs_%d-d_%.4f' % \
        (args.nlayers, args.pca_size, args.hidden_size, args.dropout)
    return fpath


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument('--language', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument("--representation", type=str, required=True)
    # Model
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--pca-size', type=int, default=300)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=.33)
    # Optimization
    parser.add_argument('--eval-batches', type=int, default=100)
    parser.add_argument('--wait-epochs', type=int, default=20)
    # Others
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=20)

    args = parser.parse_args()
    args.wait_iterations = args.wait_epochs * args.eval_batches
    args.save_path = '%s/%s/%s/%s/%s' % \
        (args.checkpoint_path, args.task, args.language, args.representation, get_model_name(args))

    util.config(args.seed)
    print(args)

    if args.representation == 'bert':
        args.pca_size = 768
    elif args.representation == 'fast':
        args.pca_size = 300
    if args.task == 'dep_label':
        args.pca_size = args.pca_size * 2
    return args


def get_model(n_classes, n_words, args):
    mlp = MLP(
        args.task, embedding_size=args.pca_size, n_classes=n_classes, hidden_size=args.hidden_size,
        nlayers=args.nlayers, dropout=args.dropout, representation=args.representation, n_words=n_words)

    if torch.cuda.device_count() > 1:
        mlp = TransparentDataParallel(mlp)
    return mlp.to(device=constants.device)


def _evaluate(evalloader, model):
    criterion = nn.CrossEntropyLoss() \
        .to(device=constants.device)

    dev_loss, dev_acc = 0, 0
    for x, y in evalloader:
        loss, acc = model.eval_batch(x, y)
        dev_loss += loss
        dev_acc += acc

    n_instances = len(evalloader.dataset)
    return {
        'loss': dev_loss / n_instances,
        'acc': dev_acc / n_instances
    }


def evaluate(evalloader, model):
    model.eval()
    with torch.no_grad():
        result = _evaluate(evalloader, model)
    model.train()
    return result


def train_epoch(trainloader, devloader, model, optimizer, criterion, train_info):
    for x, y in trainloader:
        loss = model.train_batch(x, y, optimizer, criterion)
        train_info.new_batch(loss)

        if train_info.eval:
            dev_results = evaluate(devloader, model)

            if train_info.is_best(dev_results):
                model.set_best()
            elif train_info.finish:
                train_info.print_progress(dev_results)
                return

            train_info.print_progress(dev_results)


def train(trainloader, devloader, model, eval_batches, wait_iterations):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss() \
        .to(device=constants.device)

    with tqdm(total=wait_iterations) as pbar:
        train_info = TrainInfo(pbar, wait_iterations, eval_batches)
        while not train_info.finish:
            train_epoch(trainloader, devloader, model,
                        optimizer, criterion, train_info)

    model.recover_best()


def eval_all(model, trainloader, devloader, testloader):
    train_results = evaluate(trainloader, model)
    dev_results = evaluate(devloader, model)
    test_results = evaluate(testloader, model)

    print('Final loss. Train: %.4f Dev: %.4f Test: %.4f' %
          (train_results['loss'], dev_results['loss'], test_results['loss']))

    print('Final acc. Train: %.4f Dev: %.4f Test: %.4f' %
          (train_results['acc'], dev_results['acc'], test_results['acc']))

    return train_results, dev_results, test_results


def save_results(model, train_results, dev_results, test_results, results_fname):
    results = [['n_classes', 'embedding_size', 'hidden_size', 'nlayers',
                'dropout_p', 'train_loss', 'dev_loss', 'test_loss',
                'train_acc', 'dev_acc', 'test_acc']]
    results += [[model.n_classes, model.embedding_size, model.hidden_size,
                 model.nlayers, model.dropout_p,
                 train_results['loss'], dev_results['loss'], test_results['loss'],
                 train_results['acc'], dev_results['acc'], test_results['acc']]]
    util.write_csv(results_fname, results)


def save_checkpoints(model, train_results, dev_results, test_results, save_path):
    util.mkdir(save_path)
    model.save(save_path)
    results_fname = save_path + '/results.csv'
    save_results(model, train_results, dev_results, test_results, results_fname)


def main():
    args = get_args()

    trainloader, devloader, testloader, n_classes, n_words = \
        get_data_loaders(args.data_path, args.task, args.language, args.representation, args.pca_size, args.batch_size)
    print('Language: %s Train size: %d Dev size: %d Test size: %d' %
          (args.language, len(trainloader.dataset),
           len(devloader.dataset), len(testloader.dataset)))
    print(args)

    model = get_model(n_classes, n_words, args)
    train(trainloader, devloader, model, args.eval_batches, args.wait_iterations)

    train_results, dev_results, test_results = eval_all(
        model, trainloader, devloader, testloader)

    save_checkpoints(model, train_results, dev_results, test_results, args.save_path)


if __name__ == '__main__':
    main()
