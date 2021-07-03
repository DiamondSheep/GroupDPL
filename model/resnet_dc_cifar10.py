import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict

__all__ = ['resnet20_dc', 'resnet32_dc', 'resnet44_dc', 'resnet56_dc', 'resnet110_dc', 'resnet1202_dc']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class decom_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, Dictsize, stride=1, padding=0, bias=True):
        super(decom_conv, self).__init__()
        self.calcu = nn.Sequential(
                     nn.Conv2d(in_channels, Dictsize, kernel_size=kernel_size, stride=stride, 
                               padding=padding, bias=bias), #kernel_size // 2
                     nn.Conv2d(Dictsize, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
                )
    def forward(self, x):
        return self.calcu(x)

class decom_linear(nn.Module):
    def __init__(self, in_channels, out_channels, Dictsize, bias=True):
        super(decom_linear, self).__init__()
        self.calcu = nn.Sequential(
                     nn.Linear(in_channels, Dictsize, bias=bias),
                     nn.Linear(Dictsize, out_channels, bias=bias)
        )
    def forward(self, x):
        return self.calcu(x)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, Dictsize, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        #self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = decom_conv(in_planes, planes, kernel_size=3, Dictsize=Dictsize, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = decom_conv(in_planes, planes, kernel_size=3, Dictsize=Dictsize, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     #nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     decom_conv(in_planes, planes, kernel_size=3, Dictsize=Dictsize, stride=stride, padding=1, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_dc(nn.Module):
    def __init__(self, block, num_blocks, wordconfig, num_classes=10):
        super(ResNet_dc, self).__init__()
        self.in_planes = 16
        self.wordconfig = wordconfig
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer = 1
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        #self.linear = nn.Linear(64, num_classes)
        self.linear = decom_linear(64, num_classes, self.wordconfig[self.layer])
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, Dictsize=self.wordconfig[self.layer], stride=stride))
            self.in_planes = planes * block.expansion
            self.layer += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20_dc(pretrained, wordconfig, **kwargs):
    model = ResNet_dc(BasicBlock, [3, 3, 3], wordconfig, **kwargs)
    if pretrained:
        state_dict = torch.load("./model_path/resnet20-12fca82f.th")['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.' in k:
                k = k.replace('module.', '')
            new_state_dict[k]=v
        model.load_state_dict(new_state_dict)
    return model


def resnet32_dc(pretrained, wordconfig, **kwargs):
    model = ResNet_dc(BasicBlock, [5, 5, 5], wordconfig, **kwargs)
    if pretrained:
        state_dict = torch.load("./model_path/resnet32-d509ac18.th")['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.' in k:
                k = k.replace('module.', '')
            new_state_dict[k]=v
        model.load_state_dict(new_state_dict)
    return model

def resnet44_dc(wordconfig):
    return ResNet_dc(BasicBlock, [7, 7, 7], wordconfig)


def resnet56_dc(pretrained, wordconfig, **kwargs):
    model = ResNet_dc(BasicBlock, [9, 9, 9], wordconfig, **kwargs)
    if pretrained:
        state_dict = torch.load("./model_path/resnet56-4bfd9763.th")['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.' in k:
                k = k.replace('module.', '')
            new_state_dict[k]=v
        model.load_state_dict(new_state_dict)
    return model


def resnet110_dc(wordconfig):
    return ResNet_dc(BasicBlock, [18, 18, 18], wordconfig)


def resnet1202_dc(wordconfig):
    return ResNet_dc(BasicBlock, [200, 200, 200], wordconfig)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()