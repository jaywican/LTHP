# Reference : torchvision.models.resnet.py (torchvision 0.4.1)
# Reference : https://github.com/karfly/learnable-triangulation-pytorch

import os

import torch
import torch.nn as nn

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
'''

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Confidence(nn.Module):
    '''
    Confidence = 2 conv layers + global average pooling + 3 fully-connected layers
    '''
    def __init__(self, in_channels, num_classes):
        super(Confidence, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=0.1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )
        
        self.fcs = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.convs(x)
        batch, n_channels = x.shape[:2]
        x = x.view((batch, n_channels, -1))
        x = x.mean(dim=-1)
        out = self.fcs(x)
        return out


class Backbone(nn.Module):
    '''
    2Dbackbone = ResNet152 + transposed convolution layers + 1x1 conv layer
    
    B' = Batch_size x Number of Camera(=4) = BxC
    B = number of batches
    C = number of cameras
    J = number of joints
    H, W = height, width of images

    input : torch tensor
        2D cropped images      (size: B', 3, H, W)
            or
        2D RGB images          (size: B', 3, H, W)
    output : torch tensor
        interpretable_heatmaps (size: B', J=17, H/4, W/4)
        intermedaite_heatmaps  (size: B', 256,  H/4, W/4)
        alg_confidences        (size: B', J=17)
        vol_confidences        (size: B', 32)
    '''

    def __init__(self, block, layers, num_classes=17, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, alg_confidences=False, vol_confidences=False):
        super(Backbone, self).__init__()

        self.num_joints = num_classes
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        if alg_confidences:
            self.alg_confidences = Confidence(512 * block.expansion, self.num_joints)

        if vol_confidences:
            self.vol_confidences = Confidence(512 * block.expansion, 32)

        self.transconv_layer = self._make_transconv_layer()
        self.conv1x1 = conv1x1(in_planes=256, out_planes=self.num_joints, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                #elif isinstance(m, BasicBlock):
                    #nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_transconv_layer(self, num_layers=3, num_filters=(256, 256, 256),
                              num_kernels=(4, 4, 4)):
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = 4, 1, 0
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        alg_confidences = None
        if hasattr(self, "alg_confidences"):
            alg_confidences = self.alg_confidences(x)

        vol_confidences = None
        if hasattr(self, "vol_confidences"):
            vol_confidences = self.vol_confidences(x)
        
        x = self.transconv_layer(x)
        intermediate_heatmaps = x

        x = self.conv1x1(x)
        interpretable_heatmaps = x

        return interpretable_heatmaps, intermediate_heatmaps, alg_confidences, vol_confidences


def get_backbone(args, alg_confidences=True, vol_confidences=False):
    model = Backbone(Bottleneck, [3, 8, 36, 3], num_classes=args.num_joints,
                     alg_confidences=alg_confidences, vol_confidences=vol_confidences)
    #if args.load_model:
        #print("Loading 2d backbone's pretrained weights")
        #model_state_dict = model.state_dict()
        #pretrained_state_dict = torch.load(args.ckpt_path, map_location=device)

    return model