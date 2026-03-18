import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, padding_mode="zeros"):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode)
        )

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h



class BasicMotionEncoder_U(nn.Module):
    def __init__(self, args, enable_mhs):
        super(BasicMotionEncoder_U, self).__init__()
        self.enable_mhs = enable_mhs
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192 , 128-2, 3, padding=1)

    def forward(self, flow, corr, uncertainty_):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))

        return torch.cat([out, flow], dim=1)
    
class BasicUpdateBlock_U(nn.Module):
    def __init__(self, args, hidden_dim=128, enable_mhs = False):
        super(BasicUpdateBlock_U, self).__init__()
        self.args = args
        self.enable_mhs = enable_mhs
        self.encoder = BasicMotionEncoder_U(args, enable_mhs=self.enable_mhs)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)

        self.conv_flow_head= nn.Sequential(
            conv(hidden_dim, 64),
            conv(64, 32),
            conv(32, 2, isReLU=False)
            )
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0)
            )
        
        self.uncertainty_head = nn.Sequential(
                conv(hidden_dim, 128),
                # conv(128, 128),
                conv(128, 1, isReLU=False)
                )

        self.conv_flow= nn.Sequential(
            conv(2+2+1, 64),
            conv(64, 32),
            conv(32, 2, isReLU=False)
            )


    def forward(self, net, inp_img, corr, flow_last, uncertainty = None):
        motion_features = self.encoder(flow_last, corr, uncertainty)
        inp = torch.cat([inp_img, motion_features], dim=1)

        net = self.gru(net, inp)
        flow = self.conv_flow_head(net)
        mask = self.mask(net)   #  .detach()

        # # our full model
        uncertainty = self.uncertainty_head(net)
        weight = torch.sigmoid(-uncertainty).detach()
        flow_w = flow * weight
        flow = self.conv_flow(torch.cat([flow, flow_w, uncertainty.detach()], dim=1))

        # # # unscaled flow feature and a dummy zero-map
        # uncertainty = self.uncertainty_head(net)
        # flow = self.conv_flow(torch.cat([flow, flow, torch.zeros_like(uncertainty)], dim=1))

        # # # baseline, dummy  uncertainty
        # uncertainty = torch.ones_like(flow[:, :1, :, :])*-1000.0
        
        return net, mask, flow, uncertainty