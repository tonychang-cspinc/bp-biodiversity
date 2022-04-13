import horovod.torch as hvd
import torch
import torch.nn.functional as F

EPS = 1e-7

class R2Score:
    # Adapted from https://pytorch.org/ignite/v0.4.4.post1/_modules/ignite/contrib/metrics/regression/r2_score.html
    # To support multiple regression and multiple class R^2 (and not rely on ignite)

    def __init__(self, dims, num_classes):
        self.dims = dims
        self.num_classes = num_classes
        self._num_examples = torch.zeros(num_classes)
        self._sum_of_errors = torch.zeros(num_classes, dims)
        self._y_sq_sum = torch.zeros(num_classes, dims)
        self._y_sum = torch.zeros(num_classes, dims)

    def update(self, y_pred, y, classes):
        cls_hot = torch.transpose(F.one_hot(classes, num_classes=self.num_classes), 0, 1).float()
        self._y_sum += torch.matmul(cls_hot, y).cpu().float()
        self._y_sq_sum += torch.matmul(cls_hot, torch.pow(y, 2)).cpu().float()
        self._sum_of_errors += torch.matmul(cls_hot, torch.pow(y_pred - y, 2)).cpu().float()
        self._num_examples += cls_hot.sum(1).cpu().float()

    def compute(self):
        num_examples = hvd.allreduce(self._num_examples, name="num_examples", op=hvd.Sum)
        sum_of_errors = hvd.allreduce(self._sum_of_errors, name="sum_of_errors", op=hvd.Sum)
        y_sq_sum = hvd.allreduce(self._y_sq_sum, name="y_sq_sum", op=hvd.Sum)
        y_sum = hvd.allreduce(self._y_sum, name="y_sum", op=hvd.Sum)

        #per-class R^2
        per_class = torch.ones(self.num_classes, self.dims) - sum_of_errors / (EPS + y_sq_sum - (torch.pow(y_sum, 2)) / num_examples.unsqueeze(1))

        #overall R^2
        overall = torch.ones(self.dims) - sum_of_errors.sum(0) / (y_sq_sum.sum(0) - (torch.pow(y_sum.sum(0), 2)) / num_examples.sum())
        return overall, per_class
