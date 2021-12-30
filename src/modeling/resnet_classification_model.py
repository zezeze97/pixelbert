import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.resnet50gcb_extra import ResNet50Extra
from src.modeling.resnet31gcb_extra import ResNet31Extra

class ResNetForClassification(nn.Module):
    def __init__(self, net_type = 'R50', num_class = 1000, spacial_size = (768, 768), gcb_config = None):
        super(ResNetForClassification, self).__init__()
        self.net_type = net_type
        self.num_class = num_class
        self.spacial_size = spacial_size
        self.gcb_config = gcb_config
        if self.net_type == 'R50':
            self.backbone = ResNet50Extra(layers=[3,4,6,3], input_dim=3, gcb_config=self.gcb_config, input_format="RGB")
        elif self.net_type == 'R31':
            self.backbone = ResNet31Extra(layers=[1,2,5,3], input_dim=3, gcb_config=self.gcb_config, input_format="RGB")
        else:
            print("The type of Resnet must be R50 or R31!")
        assert self.spacial_size[0] % 32 == 0 and self.spacial_size[0] % 32 == 0
        self.avgpool=nn.AvgPool2d(kernel_size = (int(self.spacial_size[0]/32),int(self.spacial_size[1]/32)))
        if self.net_type == 'R50':
            self.fc=nn.Linear(2048,num_class)
        if self.net_type == 'R31':
            self.fc=nn.Linear(512,num_class)
    def forward(self,x):
        x = self.backbone(x)[-2]
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

# test
if __name__ == '__main__':
    device = torch.device("cuda")
    gcb_config = dict(
            ratio=0.0625,
            headers=1,
            att_scale=False,
            fusion_type="channel_add",
            layers=[False, True, True, True])
    net = ResNetForClassification(net_type = 'R50', num_class = 1000, spacial_size = (768, 768), gcb_config = gcb_config).to(device)
    input = torch.randn(4,3,768,768).to(device)
    output = net(input)
    print(output.shape)


        
        
            
        

