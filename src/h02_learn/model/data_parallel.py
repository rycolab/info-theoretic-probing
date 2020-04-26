import torch.nn as nn


class TransparentDataParallel(nn.DataParallel):

    def set_best(self, *args, **kwargs):
        return self.module.set_best(*args, **kwargs)

    def recover_best(self, *args, **kwargs):
        return self.module.recover_best(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self.module.save(*args, **kwargs)

    def train_batch(self, *args, **kwargs):
        return self.module.train_batch(*args, **kwargs)

    def eval_batch(self, *args, **kwargs):
        return self.module.eval_batch(*args, **kwargs)
