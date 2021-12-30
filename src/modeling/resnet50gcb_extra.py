import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.context_block import ContextBlock
# from context_block import ContextBlock




class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, gcb_config=None):
        super(BasicBlock, self).__init__()
        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm2d(planes)

        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)

        self.conv3=nn.Conv2d(planes,planes*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(planes*self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.gcb_config = gcb_config

        if self.gcb_config is not None:
            gcb_ratio = gcb_config['ratio']
            gcb_headers = gcb_config['headers']
            att_scale = gcb_config['att_scale']
            fusion_type = gcb_config['fusion_type']
            self.context_block = ContextBlock(inplanes=planes*self.expansion,
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
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        

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


class ResNet50Extra(nn.Module):

    def __init__(self, layers, input_dim=3, gcb_config=None, input_format="BGR"):
        self.inplanes = 64
        super(ResNet50Extra, self).__init__()
        self.input_format = input_format

        self.conv1=nn.Conv2d(input_dim,self.inplanes,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.inplanes)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        
        self.stage1=self._make_layer(BasicBlock,64,layers[0],stride=1,gcb_config=get_gcb_config(gcb_config, 0))
        self.stage2=self._make_layer(BasicBlock,128,layers[1],stride=2,gcb_config=get_gcb_config(gcb_config, 1))
        self.stage3=self._make_layer(BasicBlock,256,layers[2],stride=2,gcb_config=get_gcb_config(gcb_config, 2))
        self.stage4=self._make_layer(BasicBlock,512,layers[3],stride=2,gcb_config=get_gcb_config(gcb_config, 3))


        self.grid_encoder = nn.Sequential(
            nn.Conv2d(2048, 768, kernel_size=3, stride=1, padding=1, bias=False),
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
        Return: f [(B, 2048, H/32, W/32),(B, H/64, W/64, C = 768)]
        
        '''
        f = []
        if self.input_format == "BGR":
            # RGB->BGR, images are read in as RGB by default
            x = x[:, [2, 1, 0], :, :]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #  (B, 64, H/2, W/2)
        x = self.maxpool(x)
        # (B, 64, H/4, W/4)

        x = self.stage1(x)
        # (B, 256, H/4, W/4)
        x = self.stage2(x)
        # (B, 512, H/8, W/8)
        x = self.stage3(x)
        # (B, 1024, H/16, W/16)
        x = self.stage4(x)
        # (B, 2048, H/32, W/32)
        f.append(x)

        x = self.grid_encoder(x)
        # (B, 768, H/64, W/64)
        x = x.permute(0, 2, 3, 1)
        # (B, H/64, W/64, C = 768)
        f.append(x)

        return f

    


















