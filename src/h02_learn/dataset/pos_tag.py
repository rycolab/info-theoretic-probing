import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset

from h01_data.process import get_data_file_base as get_file_names
from util import constants
from util import util


class PosTagDataset(Dataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(self, data_path, language, representation, embedding_size, mode, pca=None, classes=None, words=None):
        self.data_path = data_path
        self.language = language
        self.mode = mode
        self.representation = representation
        self.embedding_size = embedding_size

        self.input_name_base = get_file_names(data_path, language)
        self.process(pca, classes, words)

        assert self.x.shape[0] == self.y.shape[0]
        self.n_instances = self.x.shape[0]

    def process(self, pca, classes, words):
        if self.representation in ['bert', 'fast']:
            self._process(pca, classes)
            self.words = words
            self.n_words = None
        else:
            self._process_index(classes, words)
            self.pca = pca

    def _process_index(self, classes, words):
        x_raw, y_raw = self.load_data_index()

        self.load_index(x_raw, words=words)
        self.load_classes(y_raw, classes=classes)

    def load_data_index(self):
        data_ud = util.read_data(self.input_name_base % (self.mode, 'ud'))

        x_raw, y_raw = [], []
        for sentence_ud, words in data_ud:
            for i, token in enumerate(sentence_ud):
                pos_tag = token['pos']

                if pos_tag == "_" or pos_tag == "X":
                    continue

                x_raw += [words[i]]
                y_raw += [pos_tag]

        x_raw = np.array(x_raw)
        y_raw = np.array(y_raw)

        return x_raw, y_raw

    def load_index(self, x_raw, words=None):
        if words is None:
            # import ipdb; ipdb.set_trace()
            x, words = pd.factorize(x_raw, sort=True)
        else:
            new_words = set(x_raw) - set(words)
            if new_words:
                words = np.concatenate([words, list(new_words)])

            words_dict = {word: i for i, word in enumerate(words)}
            x = np.array([words_dict[token] for token in x_raw])

        self.x = torch.from_numpy(x)
        self.words = words

        self.n_words = len(words)

    def _process(self, pca, classes):
        x_raw, y_raw = self.load_data()

        self.load_embeddings(x_raw, pca=pca)
        self.load_classes(y_raw, classes=classes)

    def load_data(self):
        data_ud = util.read_data(self.input_name_base % (self.mode, 'ud'))
        data_embeddings = util.read_data(self.input_name_base % (self.mode, self.representation))

        x_raw, y_raw = [], []
        for (sentence_ud, words), (sentence_emb, _) in zip(data_ud, data_embeddings):
            for i, token in enumerate(sentence_ud):
                pos_tag = token['pos']

                if pos_tag == "_" or pos_tag == "X":
                    continue

                x_raw += [sentence_emb[i]]
                y_raw += [pos_tag]

        x_raw = np.array(x_raw)
        y_raw = np.array(y_raw)

        return x_raw, y_raw

    def load_embeddings(self, x_raw, pca=None):
        pca_x = x_raw
        self.assert_size(pca_x)

        self.x = torch.from_numpy(pca_x)
        self.pca = pca

    def assert_size(self, x):
        assert len(x[0]) == self.embedding_size

    def load_classes(self, y_raw, classes=None):
        if self.mode != 'train':
            assert classes is not None

        if classes is None:
            y, classes = pd.factorize(y_raw, sort=True)
        else:
            new_classes = set(y_raw) - set(classes)
            if new_classes:
                classes = np.concatenate([classes, list(new_classes)])

            classes_dict = {pos_class: i for i, pos_class in enumerate(classes)}
            y = np.array([classes_dict[token] for token in y_raw])

        self.y = torch.from_numpy(y)
        self.classes = classes

        self.n_classes = classes.shape[0]

    def __len__(self):
        return self.n_instances

    def __getitem__(self, index):
        return (self.x[index], self.y[index])
