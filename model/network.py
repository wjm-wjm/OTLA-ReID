import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from .backbone.resnet import resnet50


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
        init.zeros_(m.bias.data)
    elif classname.find("BatchNorm1d") != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias is not None:
            init.zeros_(m.bias.data)


class gradientreverselayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        # this is necessary. if we just return "input", "backward" will not be called sometimes
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs


class AdversarialLayer(nn.Module):
    def __init__(self, per_add_iters, iter_num=0, alpha=10.0, low_value=0.0, high_value=1.0, max_iter=10000.0):
        super(AdversarialLayer, self).__init__()
        self.per_add_iters = per_add_iters
        self.iter_num = iter_num
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter
        self.grl = gradientreverselayer.apply

    def forward(self, input, train_set=True):
        if train_set:
            self.iter_num += self.per_add_iters
        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                        self.high_value - self.low_value) + self.low_value)

        return self.grl(self.coeff, input)


class DiscriminateNet(nn.Module):
    def __init__(self, input_dim, class_num=1):
        super(DiscriminateNet, self).__init__()
        self.ad_layer1 = nn.Linear(input_dim, input_dim//2)
        self.ad_layer2 = nn.Linear(input_dim//2, input_dim//2)
        self.ad_layer3 = nn.Linear(input_dim//2, class_num)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm1d(input_dim // 2)
        self.bn2.bias.requires_grad_(False)
        self.bn = nn.BatchNorm1d(class_num)
        self.bn.bias.requires_grad_(False)
        self.sigmoid = nn.Sigmoid()

        self.ad_layer1.apply(weights_init_kaiming)
        self.ad_layer2.apply(weights_init_kaiming)
        self.ad_layer3.apply(weights_init_classifier)
        self.bn2.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        x = self.bn(x)
        x = self.sigmoid(x)  # binary classification

        return x


class BaseResNet(nn.Module):
    def __init__(self, pool_dim, class_num, per_add_iters, arch="resnet50"):
        super(BaseResNet, self).__init__()

        if arch == "resnet50":
            network = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)

        self.layer0 = nn.Sequential(network.conv1,
                                    network.bn1,
                                    network.relu,
                                    network.maxpool)
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3
        self.layer4 = network.layer4

        self.bottleneck_0 = nn.BatchNorm1d(64)
        self.bottleneck_0.bias.requires_grad_(False)  # no shift
        self.bottleneck_1 = nn.BatchNorm1d(256)
        self.bottleneck_1.bias.requires_grad_(False)  # no shift
        self.bottleneck_2 = nn.BatchNorm1d(512)
        self.bottleneck_2.bias.requires_grad_(False)  # no shift
        self.bottleneck_3 = nn.BatchNorm1d(1024)
        self.bottleneck_3.bias.requires_grad_(False)  # no shift
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.adnet = AdversarialLayer(per_add_iters=per_add_iters)
        self.disnet = DiscriminateNet(64 + 256 + 512 + 1024 + pool_dim, 1)

        self.bottleneck_0.apply(weights_init_kaiming)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.l2norm = Normalize(2)

    def forward(self, x_rgb, x_ir, modal=0, train_set=True):
        if modal == 0:
            x = torch.cat((x_rgb, x_ir), dim=0)
        elif modal == 1:
            x = x_rgb
        elif modal == 2:
            x = x_ir

        x_0 = self.layer0(x)
        x_1 = self.layer1(x_0)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        x_pool_0 = self.avgpool(x_0)
        x_pool_0 = x_pool_0.view(x_pool_0.size(0), x_pool_0.size(1))
        x_pool_1 = self.avgpool(x_1)
        x_pool_1 = x_pool_1.view(x_pool_1.size(0), x_pool_1.size(1))
        x_pool_2 = self.avgpool(x_2)
        x_pool_2 = x_pool_2.view(x_pool_2.size(0), x_pool_2.size(1))
        x_pool_3 = self.avgpool(x_3)
        x_pool_3 = x_pool_3.view(x_pool_3.size(0), x_pool_3.size(1))
        x_pool_4 = self.avgpool(x_4)
        x_pool_4 = x_pool_4.view(x_pool_4.size(0), x_pool_4.size(1))

        feat_0 = self.bottleneck_0(x_pool_0)
        feat_1 = self.bottleneck_1(x_pool_1)
        feat_2 = self.bottleneck_2(x_pool_2)
        feat_3 = self.bottleneck_3(x_pool_3)
        feat_4 = self.bottleneck(x_pool_4)

        if self.training:
            feat = torch.cat((feat_0, feat_1, feat_2, feat_3, feat_4), dim=1)
            x = self.adnet(feat, train_set=train_set)
            x_dis = self.disnet(x)
            p_4 = self.classifier(feat_4)

            return x_pool_4, p_4, x_dis
        else:
            return self.l2norm(feat_4)