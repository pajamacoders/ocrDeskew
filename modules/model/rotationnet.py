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

class BasicBlock(nn.Module):
    def __init__(self, inch: int, outch: int, kernel: int, enable_residual=False):
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential(
            ConvBnRelu(inch,outch,kernel,padding=kernel//2),
            ConvBnRelu(outch,outch,kernel,padding=kernel//2),
            nn.MaxPool2d(2,2)
        )
        self.enable_residual=enable_residual

    def forward(self, x):
        out = self.layer(x)
        if self.enable_residual:
            out+=torch.nn.functional.interpolate(x,scale_factor=0.5,mode='bilinear')
        return out

class RotationNet(nn.Module):
    def __init__(self, pretrained=None):
        super(RotationNet, self).__init__()
        self.blocks = torch.nn.Sequential(BasicBlock(1,16,5), #256x256
            BasicBlock(16,32,5), #128x128
            BasicBlock(32,64,3), #64x64
            BasicBlock(64,64,3,True) #32x32
        ) 
        self.fc = nn.Sequential(nn.Linear(64,64),nn.BatchNorm1d(64),nn.ReLU(inplace=True),nn.Linear(64,1))
        self.__init_weight()
        if pretrained:
            self.load_weight(pretrained)

    def forward(self, x):
        out = self.blocks(x)
        out = out.sum(-1).sum(-1)
        out = self.fc(out)
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