import torch
import torch.nn as nn
import ptwt
from einops import rearrange
from einops.layers.torch import Rearrange


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=None, bias=False, bn=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=bias)
        self.bn = nn.BatchNorm2d(c2) if bn is True else nn.Identity()
        self.act = act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        return self.act(self.bn(out))


class QuantConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, xscale: int=30):
        super(QuantConv2d, self).__init__()
        self.wscale = 0
        self.xscale = xscale
        self.weight = nn.Parameter(conv.weight.data.clone(), conv.weight.requires_grad)
        self.bias = nn.Parameter(conv.bias.data.clone(), conv.bias.requires_grad) if conv.bias is not None else None
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.oc = conv.out_channels
        self.register_buffer("qweight", torch.zeros_like(self.weight, dtype=torch.int8))
    
    def forward(self, x):
        xmax = max(-torch.min(x), torch.max(x)).to(torch.float32)
        wmax = max(-torch.min(self.weight), torch.max(self.weight)).to(torch.float32)
        xscale = 14 - math.floor(torch.log2(xmax)) if xmax >= 2 ** -126 else 0
        wscale = 6 - math.floor(torch.log2(wmax)) if wmax >= 2 ** -126 else 0
        self.xscale = xscale
        
        x = (torch.round(x.to(torch.float32) * (2 ** xscale)) / (2 ** xscale)).to(x.dtype)
        self.weight.data = (torch.round(self.weight.to(torch.float32) * (2 ** wscale)) / (2 ** wscale)).to(self.weight.dtype)
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return out
    
    def dforward(self, x):
        self.input_size = x.shape
        qx = torch.round(x.to(torch.float32) * (2 ** self.xscale))
        qx = torch.where(qx <= 32767., qx, torch.tensor(32767., dtype=torch.float32, device=qx.device).expand_as(qx))
        qx = torch.where(qx >= -32768., qx, torch.tensor(-32768., dtype=torch.float32, device=qx.device).expand_as(qx))
        qx = (qx / (2 ** self.xscale)).to(x.dtype)
        
        weight = (self.qweight.to(torch.float32) / (2 ** self.wscale)).to(x.dtype)
        out = F.conv2d(qx, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        return out


class BNeck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(BNeck, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                Conv(inp, inp, 3, stride, g=inp, act=nn.ReLU6()),
                # pw-linear
                Conv(inp, oup, 1, 1))
        else:
            self.conv = nn.Sequential(
                # pw
                Conv(inp, hidden_dim, 1, 1, act=nn.ReLU6()),
                # dw
                Conv(hidden_dim, hidden_dim, 3, stride, g=hidden_dim, act=nn.ReLU6()),
                # pw-linear
                Conv(hidden_dim, oup, 1, 1))

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=nn.ReLU())
        self.cv2 = Conv(c1, c_, 1, 1, act=nn.ReLU())
        self.cv3 = Conv(2 * c_, c2, 1, act=nn.ReLU())  # optional act=FReLU(c2)
        self.m = nn.Sequential(
            Conv(c_, c_, 1, 1, act=nn.ReLU()),
            Conv(c_, c_, 3, 1, act=nn.ReLU()))

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        
        return self.cv3(torch.cat((self.m(y1)+y1, y2), 1))


class Implicit(nn.Module):
    def __init__(self, c) -> None:
        super(Implicit, self).__init__()
        self.implicit = nn.Parameter(torch.ones(1, c, 1, 1))
        nn.init.normal_(self.implicit, mean=1., std=.02)
    
    def forward(self, x):
        return x * self.implicit


class ETSFA(nn.Module):
    def __init__(self, c1, c2, s) -> None:
        super(ETSFA, self).__init__()
        self.c = c1 // 2
        
        self.cv1 = Conv(c1 // 2, c2 // 2, (5,1), 1, bn=False, bias=True)
        self.cv2 = Conv(c1 // 2, c2 // 2, (1,5), 1, bn=False, bias=True)
        
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU6()
        # self.im = Implicit(c2)
        
        self.ds = nn.MaxPool2d(kernel_size=2, stride=2) if s == 2 else nn.Identity()
        # self.r = Conv(c, c, 1, 1, act=nn.ReLU6())
    
    def forward(self, x):
        s1, s2 = self.ds(x).chunk(2, 1)
        
        out = torch.cat((self.cv1(s1), self.cv2(s2)), dim=1)
        
        return self.act(self.bn(out))


class ETSFABN(nn.Module):
    def __init__(self, c1, c2, s, expand_ratio):
        super(ETSFABN, self).__init__()
        c_ = round(c1 * expand_ratio)
        self.identity = s == 1 and c1 == c2
        
        if expand_ratio == 1:
            self.conv = ETSFA(c1, c2, s)
        else:
            self.conv = nn.Sequential(
                Conv(c1, c_, 1, 1, act=nn.ReLU6()),
                ETSFA(c_, c2, s))
    
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EDFA(nn.Module):
    def __init__(self, c1, c2, l):
        super(EDFA, self).__init__()
        
        self.conv1 = Conv(c1, c1 // 2, act=nn.ReLU())
        self.conv2 = Conv(c1, c1 // 2, act=nn.ReLU())
        
        self.s1 = nn.Sequential(
            Rearrange('b c h w -> b w h c'),
            Conv(l, l, act=False),
            Rearrange('b w h c -> b h w c'),
            Conv(l, l, (1, 3), act=nn.ReLU()),
            Rearrange('b h w c -> b c h w'))
        
        self.s2 = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            Conv(l, l, act=False),
            Rearrange('b h w c -> b w h c'),
            Conv(l, l, (1, 3), act=nn.ReLU()),
            Rearrange('b w h c -> b c h w'))
        
        self.oc = Conv(c1, c2, act=nn.ReLU())

        self.res = Conv(c1, c2, act=nn.ReLU()) if c1 != c2 else nn.Identity()
    
    def forward(self, x):
        y1 = self.s1(self.conv1(x))
        y2 = self.s2(self.conv2(x))
        
        return self.oc(torch.cat([y1, y2], dim=1)) + self.res(x)


class LSA(nn.Module):
    def __init__(self, size, dim):
        super(LSA, self).__init__()
        
        self.conv = Conv(dim, 2, 1, act = nn.ReLU())
        
        self.ww = nn.Sequential(
            Conv(2, 1, (size, 5), p=(0,2), bn=False),
            Rearrange('b 1 1 s -> b s 1'),
            nn.Conv1d(size, size, 1, 1),
            nn.Sigmoid()
        )
        
        self.hw = nn.Sequential(
            Conv(2, 1, (5, size), p=(2,0), bn=False),
            Rearrange('b 1 s 1 -> b s 1'),
            nn.Conv1d(size, size, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        bs = x.shape[0]
        
        ww = self.ww(x)
        hw = self.hw(x)
        
        return torch.cat([torch.mm(hw[i],ww[i].t()).unsqueeze(0).unsqueeze(0) for i in range(bs)], dim=0)
