
class TrainInfo:
    batch_id = 0
    running_loss = []
    best_loss = float('inf')
    best_batch = 0

    def __init__(self, pbar, wait_iterations, eval_batches):
        self.pbar = pbar
        self.wait_iterations = wait_iterations
        self.eval_batches = eval_batches

    @property
    def finish(self):
        return (self.batch_id - self.best_batch) >= self.wait_iterations

    @property
    def eval(self):
        return (self.batch_id % self.eval_batches) == 0

    @property
    def max_epochs(self):
        return self.best_batch + self.wait_iterations

    @property
    def avg_loss(self):
        return sum(self.running_loss) / len(self.running_loss)

    def new_batch(self, loss):
        self.batch_id += 1
        self.pbar.update(1)
        self.running_loss += [loss]

    def is_best(self, dev_results):
        dev_loss = dev_results['loss']
        if dev_loss < self.best_loss:
            self.best_loss = dev_loss
            self.best_batch = self.batch_id
            self.pbar.total = self.max_epochs
            return True

        return False

    def reset_loss(self):
        self.running_loss = []

    def print_progress(self, dev_results):
        dev_loss = dev_results['loss']
        dev_acc = dev_results['acc']
        self.pbar.set_description(
            'Training loss: %.4f Dev loss: %.4f acc: %.4f' %
            (self.avg_loss, dev_loss, dev_acc))
        self.reset_loss()
