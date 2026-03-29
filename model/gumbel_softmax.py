import torch


def gumbel_softmax(logits, tau, hard=False):
    gumbels = -torch.empty_like(logits).contiguous().exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim=-1)
    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).contiguous().scatter_(dim=-1, index=index, value=1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret