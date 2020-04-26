import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset

from h01_data.process import get_data_file_base as get_file_names
from util import constants
from util import util
from .pos_tag import PosTagDataset


class DepLabelDataset(PosTagDataset):
    # pylint: disable=too-many-instance-attributes

    def load_data_index(self):
        data_ud = util.read_data(self.input_name_base % (self.mode, 'ud'))

        x_raw, y_raw = [], []
        for sentence_ud, words in data_ud:
            for i, token in enumerate(sentence_ud):
                head = token['head']
                rel = token['rel']

                if rel == "_" or rel == "root":
                    continue

                x_raw_tail = words[i]
                x_raw_head = words[head - 1]

                x_raw += [[x_raw_tail, x_raw_head]]
                y_raw += [rel]

        x_raw = np.array(x_raw)
        y_raw = np.array(y_raw)

        return x_raw, y_raw

    def load_index(self, x_raw, words=None):
        if words is None:
            words = []

        new_words = sorted(list(set(np.unique(x_raw)) - set(words)))
        if new_words:
            words = np.concatenate([words, new_words])

        words_dict = {word: i for i, word in enumerate(words)}
        x = np.array([[words_dict[token] for token in tokens] for tokens in x_raw])

        self.x = torch.from_numpy(x)
        self.words = words

        self.n_words = len(words)


    def load_data(self):
        data_ud = util.read_data(self.input_name_base % (self.mode, 'ud'))
        data_embeddings = util.read_data(self.input_name_base % (self.mode, self.representation))

        x_raw, y_raw = [], []
        for (sentence_ud, words), (sentence_emb, _) in zip(data_ud, data_embeddings):
            for i, token in enumerate(sentence_ud):
                head = token['head']
                rel = token['rel']

                if rel == "_" or rel == "root":
                    continue

                x_raw_tail = sentence_emb[i]
                x_raw_head = sentence_emb[head - 1]

                x_raw += [np.concatenate([x_raw_tail, x_raw_head])]
                y_raw += [rel]

        x_raw = np.array(x_raw)
        y_raw = np.array(y_raw)

        return x_raw, y_raw
