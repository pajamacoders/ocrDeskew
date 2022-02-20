import os
import torch
import torch.nn as nn


class ConvBnRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1):
        super(ConvBnRelu, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, groups,
                      False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True))

    def forward(self, x):
        return self.conv_bn_relu(x)

class DeskewNet(nn.Module):
    def __init__(self, pretrained=None):
        super(DeskewNet, self).__init__()
        k=3
        self.backbone = nn.Sequential(
            ConvBnRelu(1,8,k,padding=k//2),
            nn.MaxPool2d(2,2), #256x256
            ConvBnRelu(8,16,k,padding=k//2),
            nn.MaxPool2d(2,2), #128x128
            ConvBnRelu(16,32,k,padding=k//2),
            nn.MaxPool2d(2,2), #64x64
            ConvBnRelu(32,64,k,padding=k//2),
            nn.MaxPool2d(2,2), #32x32
            ConvBnRelu(64,64,k,padding=k//2),
            nn.MaxPool2d(2,2), #16x16
        )
        self.avgpool = nn.AvgPool2d((16,16))
        self.fc = nn.Sequential(nn.Linear(64,64),nn.Linear(64,1))
        self.__init_weight()
        if pretrained:
            self.load_weight(pretrained)

    def forward(self, x):
        out = self.backbone(x)
        out = self.avgpool(out)
        out = self.fc(out.squeeze())
        out = torch.rad2deg(out)
        return out

    def __init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            else: 
                pass

    def load_weight(self, pretrained):
        if os.path.exists(pretrained):
            pre_state_dict = torch.load(pretrained)['model']
            state_dict = self.state_dict()
            total_params = len(state_dict.keys())
            hit_cnt = 0
            for k, v in pre_state_dict.items():
                if k in state_dict.keys():
                    state_dict[k]=v
                    hit_cnt+=1
            print(f'hit count : {hit_cnt}/{total_params} prameter loaded!') 
            self.load_state_dict(state_dict)

                    
    def weight_init_xavier_uniform(self, sub):
        print(sub)