import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.context_block import ContextBlock
# from context_block import ContextBlock

def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """ 1x1 convolution """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, gcb_config=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.9)
        self.downsample = downsample
        self.stride = stride
        self.gcb_config = gcb_config

        if self.gcb_config is not None:
            gcb_ratio = gcb_config['ratio']
            gcb_headers = gcb_config['headers']
            att_scale = gcb_config['att_scale']
            fusion_type = gcb_config['fusion_type']
            self.context_block = ContextBlock(inplanes=planes,
                                                        ratio=gcb_ratio,
                                                        headers=gcb_headers,
                                                        att_scale=att_scale,
                                                        fusion_type=fusion_type)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.gcb_config is not None:
            out = self.context_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def get_gcb_config(gcb_config, layer):
    if gcb_config is None or not gcb_config['layers'][layer]:
        return None
    else:
        return gcb_config


class ResNet31Extra(nn.Module):

    def __init__(self, layers, input_dim=3, gcb_config=None, input_format="BGR"):
        assert len(layers) >= 4

        super(ResNet31Extra, self).__init__()
        self.input_format = input_format
        self.inplanes = 128
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer1 = self._make_layer(BasicBlock, 256, layers[0], stride=1, gcb_config=get_gcb_config(gcb_config, 0))

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer2 = self._make_layer(BasicBlock, 512, layers[1], stride=1, gcb_config=get_gcb_config(gcb_config, 1))

        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer3 = self._make_layer(BasicBlock, 512, layers[2], stride=1, gcb_config=get_gcb_config(gcb_config, 2))

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)
        
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=1, gcb_config=get_gcb_config(gcb_config, 3))

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)

        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.grid_encoder = nn.Sequential(
            nn.Conv2d(512, 768, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, gcb_config=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if blocks == 1:
            layers.append(block(self.inplanes, planes, stride, downsample, gcb_config=gcb_config))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample, gcb_config=None))
        self.inplanes = planes * block.expansion
        if blocks > 1:
            for _ in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes, gcb_config=None))
            layers.append(block(self.inplanes, planes, gcb_config=gcb_config))

        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        Args: x (B, C=3, H, W)
        Return: f [(B, 512, H/32, W/32),(B, H/64, W/64, C = 768)]
        
        '''
        f = []
        if self.input_format == "BGR":
            # RGB->BGR, images are read in as RGB by default
            x = x[:, [2, 1, 0], :, :]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        #  (B, 64, H, W)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # (B, 128, H, W)

        x = self.maxpool1(x)
        # (B, 128, H/2, W/2)
        x = self.layer1(x)
        # (B, 256, H/2, W/2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        # (B, 256, H/2, W/2)

        x = self.maxpool2(x)
        # (B, 256, H/4, W/4)
        x = self.layer2(x)
        # (B, 512, H/4, W/4)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        # (B, 512, H/4, W/4)

        x = self.maxpool3(x)
        # (B, 512, H/8, W/8)

        x = self.layer3(x)
        # (B, 512, H/8, W/8)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # (B, 512, H/8, W/8)

        x = self.maxpool4(x)
        # (B, 512, H/16, W/16)

        x = self.layer4(x)
        # (B, 512, H/16, W/16)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        # (B, 512, H/16, W/16)
        x = self.maxpool5(x)
        # (B, 512, H/32, W/32)
        f.append(x)

        x = self.grid_encoder(x)
        # (B, 768, H/64, W/64)
        x = x.permute(0, 2, 3, 1)
        # (B, H/64, W/64, C = 768)
        f.append(x)

        return f

# test
if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device("cuda")
    gcb_config=dict(
                ratio=0.0625,
                headers=1,
                att_scale=False,
                fusion_type="channel_add",
                layers=[False, True, True, True],
            )
    net = ResNet31Extra(layers=[1,2,5,3], input_dim=3, gcb_config=gcb_config, input_format="BGR").to(device)
    input = torch.randn(2,3,768,768).to(device) # (B, channel, H, W)
    output = net(input)
    print(output[0].shape,output[1].shape)
    summary(net, input_size=(3,224,224), batch_size=2)










