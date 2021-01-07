import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def cross_entropy_2d(predict, target, weight):
    """
    Args:
        predict:(n, c, h, w)
        target:(n, h, w)
    """
    assert not target.requires_grad
    assert predict.dim() == 4
    assert target.dim() == 3
    assert predict.size(0) == target.size(0), f"{predict.size(0)} vs {target.size(0)}"
    assert predict.size(2) == target.size(1), f"{predict.size(2)} vs {target.size(1)}"
    assert predict.size(3) == target.size(2), f"{predict.size(3)} vs {target.size(3)}"
    n, c, h, w = predict.size()
    target_mask = (target >= 0) * (target != 255)
    target = target[target_mask]
    if not target.data.dim():
        return Variable(torch.zeros(1))
    predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
    predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
    loss = F.cross_entropy(predict, target, weight=weight, size_average=True)
    return loss


def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


def robust_entropy_loss(v):
    P = F.softmax(v, dim=1)        # [B, 19, H, W]
    logP = F.log_softmax(v, dim=1) # [B, 19, H, W]
    PlogP = P * logP               # [B, 19, H, W]
    ent = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W]
    ent = ent / 2.9444         # chanage when classes is not 19
    # compute robust entropy
    ent = ent ** 2.0 + 1e-8
    ent = ent ** 2.0
    return ent.mean()


if __name__ == '__main__':
    # Set properties
    batch_size = 10
    out_channels = 2
    W = 10
    H = 10

    # Initialize logits etc. with random
    logits = torch.FloatTensor(batch_size, out_channels, H, W).normal_()
    target = torch.LongTensor(batch_size, H, W).random_(0, out_channels)
    weights = torch.FloatTensor(batch_size, 1, H, W).random_(1, 3)
    weights = Variable(weights)

    # Calculate log probabilities
    logp = F.log_softmax(logits)

    # Gather log probabilities with respect to target
    logp = logp.gather(1, target.view(batch_size, 1, H, W))

    # Multiply with weights
    weighted_logp = (logp * weights).view(batch_size, -1)

    # Rescale so that loss is in approx. same interval
    weighted_loss = weighted_logp.sum(1) / weights.view(batch_size, -1).sum(1)

    # Average over mini-batch
    weighted_loss = weighted_loss.mean()
