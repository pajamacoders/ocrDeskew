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

class DeskewNetV4(nn.Module):
    def __init__(self, buckets, last_fc_in_ch, pretrained=None):
        super(DeskewNetV4, self).__init__()
        buckets = eval(buckets) if isinstance(buckets, str) else buckets
        assert isinstance(buckets, int), 'buckets must be type int'
        k=5
        self.block1 = nn.Sequential(
                ConvBnRelu(1,8,k,padding=k//2),
                nn.MaxPool2d(2,2), #256x256
                ConvBnRelu(8,16,k,padding=k//2),
                nn.MaxPool2d(2,2), #128x128
            )
        
        self.block2 = ConvBnRelu(16,8,k,padding=k//2)
        self.fc = nn.Sequential(nn.Linear(131072,last_fc_in_ch, bias=False),
            nn.BatchNorm1d(last_fc_in_ch),
            nn.ReLU(True),
            nn.Linear(last_fc_in_ch, buckets, bias=False))

        self.__init_weight()
        if pretrained:
            self.load_weight(pretrained)

    def forward(self, x):
        out = self.block1(x)
        out = torch.fft.fft2(out)
        out = out.real**2+out.imag**2
        out = torch.log(1.0+out)
        out = self.block2(out)
        bs,c,h,w = out.shape
        out = out.reshape(bs,-1)#torch.flatten(tmp,1)
        out = self.fc(out)

        return out
    
    def predict(self, x):
        out = self.block1(x)
        out = torch.fft.fft2(out)
        out = out.real**2+out.imag**2
        out = torch.log(1.0+out)
        out = self.block2(out)
        bs,c,h,w = out.shape
        out = out.reshape(bs,-1)#torch.flatten(tmp,1)
        out = self.fc(out)
        return torch.argmax(torch.softmax(out, -1), -1)

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