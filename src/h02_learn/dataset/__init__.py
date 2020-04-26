import torch
from torch.utils.data import DataLoader

from util import constants
from util import util
from .pos_tag import PosTagDataset
from .dep_label import DepLabelDataset


def generate_batch(batch):
    x = torch.cat([item[0].unsqueeze(0) for item in batch], dim=0)
    y = torch.cat([item[1].unsqueeze(0) for item in batch], dim=0)

    x, y = x.to(device=constants.device), y.to(device=constants.device)
    return (x, y)


def get_data_cls(task):
    if task == 'pos_tag':
        return PosTagDataset
    if task == 'dep_label':
        return DepLabelDataset


def get_data_loader(dataset_cls, data_path, language, representations, pca_size, mode, batch_size, shuffle, pca=None, classes=None, words=None):
    trainset = dataset_cls(data_path, language, representations, pca_size, mode, pca=pca, classes=classes, words=words)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, collate_fn=generate_batch)
    return trainloader, trainset.pca, trainset.classes, trainset.words


def get_data_loaders(data_path, task, language, representations, pca_size, batch_size):
    dataset_cls = get_data_cls(task)

    trainloader, pca, classes, words = get_data_loader(
        dataset_cls, data_path, language, representations, pca_size, 'train', batch_size=batch_size, shuffle=True)
    devloader, _, classes, words = get_data_loader(
        dataset_cls, data_path, language, representations, pca_size, 'dev', batch_size=batch_size, shuffle=False, pca=pca, classes=classes, words=words)
    testloader, _, classes, words = get_data_loader(
        dataset_cls, data_path, language, representations, pca_size, 'test', batch_size=batch_size, shuffle=False, pca=pca, classes=classes, words=words)
    return trainloader, devloader, testloader, testloader.dataset.n_classes, testloader.dataset.n_words
