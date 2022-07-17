import torch
import torch.nn as nn


def normalize(x, axis=-1):
    """
    Normalizing to unit length along the specified dimension.
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class TripletLoss(nn.Module):
    """
    Triplet loss with hard positive/negative mining.
    Reference: Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
    - margin (float): margin for triplet.
    - inputs: feature matrix with shape (batch_size, feat_dim).
    - targets: ground truth labels with shape (num_classes).
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()  # torch.eq: greater than or equal to >=

        return loss, correct


class PredictionAlignmentLoss(nn.Module):
    """
    Proposed loss for Prediction Alignment Learning (PAL).
    """
    def __init__(self, lambda_vr=0.1, lambda_rv=0.5):
        super(PredictionAlignmentLoss, self).__init__()
        self.lambda_vr = lambda_vr
        self.lambda_rv = lambda_rv

    def forward(self, x_rgb, x_ir):
        sim_rgbtoir = torch.mm(normalize(x_rgb), normalize(x_ir).t())
        sim_irtorgb = torch.mm(normalize(x_ir), normalize(x_rgb).t())
        sim_irtoir = torch.mm(normalize(x_ir), normalize(x_ir).t())

        sim_rgbtoir = nn.Softmax(1)(sim_rgbtoir)
        sim_irtorgb = nn.Softmax(1)(sim_irtorgb)
        sim_irtoir = nn.Softmax(1)(sim_irtoir)

        KL_criterion = nn.KLDivLoss(reduction="batchmean")

        x_rgbtoir = torch.mm(sim_rgbtoir, x_ir)
        x_irtorgb = torch.mm(sim_irtorgb, x_rgb)
        x_irtoir = torch.mm(sim_irtoir, x_ir)

        x_rgb_s = nn.Softmax(1)(x_rgb)
        x_rgbtoir_ls = nn.LogSoftmax(1)(x_rgbtoir)
        x_irtorgb_s = nn.Softmax(1)(x_irtorgb)
        x_irtoir_ls = nn.LogSoftmax(1)(x_irtoir)

        loss_rgbtoir = KL_criterion(x_rgbtoir_ls, x_rgb_s)
        loss_irtorgb = KL_criterion(x_irtoir_ls, x_irtorgb_s)

        loss = self.lambda_vr * loss_rgbtoir + self.lambda_rv * loss_irtorgb

        return loss, sim_rgbtoir, sim_irtorgb