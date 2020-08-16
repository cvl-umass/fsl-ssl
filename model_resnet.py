import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init

# track_running_stats=False

class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out

class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out
            
class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight 
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_fw(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return Conv2d_fw(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    maml = False #Default
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, downsample_jigsaw=None, track_running_stats=True, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn

        if self.maml:
            self.conv1 = conv3x3_fw(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        if self.use_bn:
            if self.maml:
                self.bn1 = BatchNorm2d_fw(planes)
            else:
                self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        if self.maml:
            self.conv2 = conv3x3_fw(planes, planes)
        else:
            self.conv2 = conv3x3(planes, planes)
        if self.use_bn:
            if self.maml:
                self.bn2 = BatchNorm2d_fw(planes)
            else:
                self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        self.downsample = downsample
        self.downsample_jigsaw = downsample_jigsaw
        self.stride = stride


    def forward(self, x, jigsaw=False):
        residual = x

        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    maml = False #Default
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, track_running_stats=True, use_bn=True):
        super(Bottleneck, self).__init__()
        self.use_bn = use_bn

        if self.maml:
            self.conv1 = Conv2d_fw(inplanes, planes, kernel_size=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if self.use_bn:
            if self.maml:
                self.bn1 = BatchNorm2d_fw(planes)
            else:
                self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        if self.maml:
            self.conv2 = Conv2d_fw(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        if self.use_bn:
            if self.maml:
                self.bn2 = BatchNorm2d_fw(planes)
            else:
                self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
        if self.use_bn:
            self.conv3 = Conv2d_fw(planes, planes * 4, kernel_size=1, bias=False)
        else:
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        if self.use_bn:
            if self.maml:
                self.bn3 = BatchNorm2d_fw(planes * 4)
            else:
                self.bn3 = nn.BatchNorm2d(planes * 4, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.use_bn:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MyModule(nn.Module):
    maml = False #Default
    def __init__(self, layers):
        super(MyModule, self).__init__()
        self.layers = layers

    def forward(self, x, jigsaw=False):
        for _,layer in enumerate(self.layers):
            x = layer(x, jigsaw)
        return x

class ResNet(nn.Module):
    maml = False #Default
    def __init__(self, block, layers,  network_type, num_classes, att_type=None, tracking=True, use_bn=True):
        self.track_running_stats = tracking
        self.use_bn = use_bn

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR 
        if network_type == "ImageNet":
            if self.maml:
                self.conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            else:
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
            self.avgpool_jigsaw = nn.AvgPool2d(2)
        else:
            if self.maml:
                self.conv1 = Conv2d_fw(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            else:
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        if self.use_bn:
            if self.maml:
                self.bn1 = BatchNorm2d_fw(64)
            else:
                self.bn1 = nn.BatchNorm2d(64, track_running_stats=self.track_running_stats)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type, use_bn=self.use_bn)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type, use_bn=self.use_bn)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type, use_bn=self.use_bn)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type, use_bn=self.use_bn)

        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0


    def _make_layer(self, block, planes, blocks, stride=1, att_type=None, use_bn=True):
        downsample = None
        downsample_jigsaw = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if use_bn:
                if self.maml:
                    downsample = nn.Sequential(
                        Conv2d_fw(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                        BatchNorm2d_fw(planes * block.expansion),
                    )
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes * block.expansion, track_running_stats=self.track_running_stats),
                    )
            else:
                if self.maml:
                    downsample = nn.Sequential(
                        Conv2d_fw(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                    )
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride, bias=False),
                    )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, \
                            downsample_jigsaw=downsample_jigsaw, track_running_stats=self.track_running_stats, use_bn=self.use_bn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, track_running_stats=self.track_running_stats, use_bn=self.use_bn))

        return nn.Sequential(*layers)

    def forward(self, x, jigsaw=False):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.network_type == "ImageNet":
            if x.shape[-1] == 7:
                x = self.avgpool(x)
            else:
                x = self.avgpool_jigsaw(x)
        else:
            x = F.avg_pool2d(x, 4)
        return x

def ResidualNet(network_type, depth, num_classes, att_type, tracking=True, use_bn=True):
    maml = False #Default
    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type, tracking, use_bn)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type, tracking, use_bn)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type, tracking, use_bn)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type, tracking, use_bn)

    return model