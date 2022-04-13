import horovod.torch as hvd
import torch
import torch.nn.functional as F

EPS = 1e-7

class RMSE:
    # Supports multiple class RMSE

    def __init__(self, dims, num_classes, maxs=None, means=None, stds=None):
        self.dims = dims
        self.num_classes = num_classes
        self._num_examples = torch.zeros(num_classes)
        self._sse = torch.zeros(num_classes, dims).float()
        self._maxs = maxs
        self._means = means
        self._stds = stds

    def update(self, y_pred, y, classes):
        cls_hot = torch.transpose(F.one_hot(classes, num_classes=self.num_classes), 0, 1).float()
        if self._maxs is not None:
            self._sse += torch.matmul(cls_hot, torch.pow((y - y_pred) * self._maxs, 2)).cpu().float()
        elif self._means is not None:
            self._sse += torch.matmul(cls_hot, torch.pow((y - y_pred) * self._stds + self._means, 2)).cpu().float()
        else:
            self._sse += torch.matmul(cls_hot, torch.pow((y - y_pred), 2)).cpu().float()
        self._num_examples += cls_hot.sum(1).cpu().float()

    def compute(self):
        num_examples = hvd.allreduce(self._num_examples, name="num_examples", op=hvd.Sum)
        sse = hvd.allreduce(self._sse, name="sse", op=hvd.Sum)
        per_class = (sse / num_examples.unsqueeze(1)).sqrt()

        overall = (sse.sum(0) / num_examples.sum()).sqrt()

        return overall, per_class
