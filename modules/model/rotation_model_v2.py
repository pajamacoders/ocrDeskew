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

class DeskewNetV2(nn.Module):
    def __init__(self, pretrained=None):
        super(DeskewNetV2, self).__init__()
        k=3
        self.backbone = nn.Sequential(
            ConvBnRelu(1,8,k,padding=k//2, stride=2),#256x256
            ConvBnRelu(8,16,k,padding=k//2, stride=2),#128x128
            ConvBnRelu(16,32,5,padding=5//2,stride=2),#64x64
            ConvBnRelu(32,64,5,padding=5//2,stride=2),#32x32
            ConvBnRelu(64,128,5,padding=5//2,stride=2),#16x16
        )
        self.avgpool = nn.AvgPool2d((16,16))
        self.fc = nn.Sequential(nn.Linear(32768,128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128,1))
        self.__init_weight()
        if pretrained:
            self.load_weight(pretrained)

    def forward(self, x):
        out = self.backbone(x)
        factor = self.avgpool(out).softmax(1)
        out*=factor
        bs,c,h,w = out.shape
        out=out.reshape(bs,-1)
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