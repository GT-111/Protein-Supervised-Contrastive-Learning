import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(mask).scatter_(1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos
        return loss.mean()

class MultiSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, lambda_label=1.0, lambda_subclass=1.0):
        super(MultiSupConLoss, self).__init__()
        self.loss = SupConLoss(temperature)
        self.lambda_label = lambda_label
        self.lambda_subclass = lambda_subclass

    def forward(self, features, labels, subclasses):
        loss_label = self.loss(features, labels)
        loss_sub = self.loss(features, subclasses)
        return self.lambda_label * loss_label + self.lambda_subclass * loss_sub