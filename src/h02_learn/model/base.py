from abc import ABC, abstractmethod
import copy
import torch
import torch.nn as nn

from util import constants


class BaseModel(nn.Module, ABC):
    # pylint: disable=abstract-method
    name = 'base'

    def __init__(self):
        super().__init__()

        self.best_state_dict = None

    def set_best(self):
        self.best_state_dict = copy.deepcopy(self.state_dict())

    def recover_best(self):
        self.load_state_dict(self.best_state_dict)

    def save(self, path):
        fname = self.get_name(path)
        torch.save({
            'kwargs': self.get_args(),
            'model_state_dict': self.state_dict(),
        }, fname)

    @abstractmethod
    def get_args(self):
        pass

    @classmethod
    def load(cls, path):
        checkpoints = cls.load_checkpoint(path)
        model = cls(**checkpoints['kwargs'])
        model.load_state_dict(checkpoints['model_state_dict'])
        return model

    @classmethod
    def load_checkpoint(cls, path):
        fname = cls.get_name(path)
        return torch.load(fname, map_location=constants.device)

    @classmethod
    def get_name(cls, path):
        return '%s/model.tch' % (path)
