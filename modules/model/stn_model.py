import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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
class MV2Block(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class STDirNet(nn.Module):
    def __init__(self, pretrained=None):
        super(STDirNet, self).__init__()
        self.layer=nn.Sequential(
            ConvBnRelu(1,16,kernel_size=7, stride=1,padding=7//2),
            MV2Block(16,16, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            MV2Block(16,32,kernel_size=5),
            MV2Block(32,32,kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            MV2Block(32,64),
            MV2Block(64,64),
            nn.Dropout2d(0.1),

        )            
        self.fc1=nn.Sequential(nn.Linear(6400, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4)
        )
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=13, padding=13//2),
            MV2Block(16,16, kernel_size=9),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            MV2Block(16,32, kernel_size=5),
            MV2Block(32,32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            MV2Block(32,64),
            MV2Block(64,64),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 2)
        )
        if pretrained:
            self.load_weight(pretrained)
        else:
            self.init_weights(self.layer.modules())
            self.init_weights(self.fc1.modules())
            self.init_weights(self.localization.modules())
            self.init_weights(self.fc_loc.modules())

            # Initialize the weights/bias with identity transformation
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 64 * 5 * 5)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = self.layer(x)
        x = x.view(-1, 64*10*10)
        x = self.fc1(x)
        return x
    
    def predict(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = self.layer(x)
        x = x.view(-1, 64*10*10)
        x = self.fc1(x)
        prob = torch.softmax(x, -1)

        return torch.argmax(prob, -1)
    
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
    
    def init_weights(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()