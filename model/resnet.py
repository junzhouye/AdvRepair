'''
ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch.nn as nn
import torch.nn.functional as F
import torch


class Probe(nn.Module):
    """
    探针
    """

    def __init__(self, in_ch, fc_ch, layer_num=2, num_class=10):
        super(Probe, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )
        if layer_num == 2:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, in_ch * 2, 3, 2, 1),
                nn.Conv2d(in_ch * 2, in_ch * 2, 3, 2, 1),
                nn.BatchNorm2d(in_ch * 2),
                nn.ReLU(),
                nn.Conv2d(in_ch * 2, in_ch * 4, 3, 1, 1),
                nn.Conv2d(in_ch * 4, in_ch * 4, 3, 1, 1),
                nn.BatchNorm2d(in_ch * 4),
                nn.ReLU(),
                nn.AvgPool2d(4, 4)
            )
        elif layer_num == 1:
            self.convs = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, 1, 1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.Conv2d(in_ch, in_ch, 3, 1, 1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.AvgPool2d(4, 4)
            )
        self.fc = nn.Linear(fc_ch, num_class)

    def forward(self, x):
        feat = self.features(x)
        feat = self.convs(feat)
        feat = feat.view(feat.size(0), -1)

        out = self.fc(feat)
        return out


class Probe2(nn.Module):
    def __init__(self, in_ch, num_class=10):
        super(Probe2, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_ch, 1024),
            nn.ReLU(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            nn.Linear(1024, num_class)
        )

    def forward(self, x):
        out = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Probe_2L(nn.Module):
    def __init__(self, in_ch, num_class=10):
        super(Probe_2L, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_ch, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_class)
        )

    def forward(self, x):
        out = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_feature(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        return feature, out

    def get_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(out)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        out = F.avg_pool2d(f4, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return f1, f2, f3, f4, out

    def get_probe_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(out)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        out = F.avg_pool2d(f4, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return f1, f2, f3, out


def ResNet18(num_class=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_class)


def ResNet34(num_class=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_class)


def ResNet50(num_class=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_class)


def ResNet101(num_class=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_class)


def ResNet152(num_class=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_class)


class ResNet_TINY(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_TINY, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(2048 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_feature(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        return feature, out

    def get_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(out)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        out = F.avg_pool2d(f4, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return f1, f2, f3, f4, out

    def get_probe_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(out)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        out = F.avg_pool2d(f4, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return f1, f2, f3, out


def ResNet18_TINY(num_class=10):
    return ResNet_TINY(BasicBlock, [2, 2, 2, 2], num_classes=num_class)


def ResNet34_TINY(num_class=10):
    return ResNet_TINY(BasicBlock, [3, 4, 6, 3], num_classes=num_class)


def ResNet50_TINY(num_class=10):
    return ResNet_TINY(Bottleneck, [3, 4, 6, 3], num_classes=num_class)


def ResNet101_TINY(num_class=10):
    return ResNet_TINY(Bottleneck, [3, 4, 23, 3], num_classes=num_class)


def ResNet152_TINY(num_class=10):
    return ResNet_TINY(Bottleneck, [3, 8, 36, 3], num_classes=num_class)


def main():
    model = ResNet18()
    base_width = 64
    num_classes = 2
    probe1 = Probe(base_width, base_width * 16, 2, num_classes)
    probe2 = Probe(base_width * 2, base_width * 8, 2, num_classes)
    probe3 = Probe(base_width * 4, base_width * 16, 1, num_classes)
    probe4 = Probe(base_width * 8, base_width * 8, 1, num_classes)

    in_ = torch.rand(1, 3, 32, 32)
    f1, f2, f3, f4, out = model.get_features(in_)

    print(f1.size(), f2.size(), f3.size(), f4.size(), out.size())

    f1_probe = probe1(f1)
    print(f1_probe)
    f2_probe = probe2(f2)
    print(f2_probe)
    f3_probe = probe3(f3)
    print(f3_probe)
    f4_probe = probe4(f4)
    print(f4_probe)


if __name__ == "__main__":
    model = ResNet50_TINY(num_class=200)
    i = torch.randn(1, 3, 64, 64)

    a, out = model.get_feature(i)
    # print([s.size() for s in a])
    # print(out.size())

    print(a.size())
    print(out.size())
