import horovod.torch as hvd
import torch
import torch.nn.functional as F


class ClassMetrics:
    def __init__(self, num_classes):
        self.true_pos = torch.zeros(num_classes)
        self.all_pos = torch.zeros(num_classes)
        self.all_pred = torch.zeros(num_classes)
        self.num_classes = num_classes
        self.eps = 1e-20

    def update(self, y_pred, y):
        yp_hot = F.one_hot(
            torch.argmax(y_pred, dim=1),
            num_classes=self.num_classes
        ).cpu().float()
        y_hot = F.one_hot(y, num_classes=self.num_classes).cpu().float()
        correct = yp_hot * y_hot
        self.true_pos += correct.sum(dim=0)
        self.all_pos += yp_hot.sum(dim=0)
        self.all_pred += y_hot.sum(dim=0)

    def compute(self):
        true_pos = hvd.allreduce(self.true_pos, name="tps", op=hvd.Sum)
        all_pos = hvd.allreduce(self.all_pos, name="all_pos", op=hvd.Sum)
        all_pred = hvd.allreduce(self.all_pred, name="all_pred", op=hvd.Sum)
        precisions = true_pos / (all_pos + self.eps)
        recalls = true_pos / (all_pred + self.eps)
        f1s = 2 * (precisions * recalls) / (precisions + recalls + self.eps)
        return precisions, recalls, f1s
