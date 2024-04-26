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


# class ETSFABN(nn.Module):
#     def __init__(self, c1, c2, s) -> None:
#         super(ETSFABN, self).__init__()
#         c_ = c2 // 2
        
#         self.ds = nn.MaxPool2d(kernel_size=2, stride=2) if s == 2 else nn.Identity()
        
#         self.cv1 = Conv(c1, c_, 1, 1, act=nn.ReLU())
#         self.cv2 = Conv(c1, c_, 1, 1, act=nn.ReLU())
#         self.cout = Conv(4*c_, c2, 1, 1, act=nn.ReLU())
        
#         self.etsfa1 = nn.Sequential(
#             Conv(c_, c_, (3,1), 1, bn=False),
#             Implicit(c_),
#             Conv(c_, c_, (1,3), 1, act=nn.ReLU()))

#         self.etsfa2 = nn.Sequential(
#             Conv(c_, c_, (3,1), 1, bn=False),
#             Implicit(c_),
#             Conv(c_, c_, (1,3), 1, act=nn.ReLU()))
    
#     def forward(self, x):        
#         x = self.ds(x)
        
#         y1 = self.cv1(x)
#         y2 = self.cv2(x)
#         y3 = self.etsfa1(y2)
#         y4 = self.etsfa2(y3)

#         return self.cout(torch.cat((y1, y2, y3, y4), dim=1))


class ETSFABN(nn.Module):
    def __init__(self, c1, c2, s):
        super(ETSFABN, self).__init__()
        c_ = round(c1 * s * 2)
        self.identity = s == 1 and c1 == c2
        
        self.conv = nn.Sequential(
            Conv(c1, c_, (3,1), (s,1), bn=False),
            Implicit(c_),
            Conv(c_, c_, (1,3), (1,s), act=nn.ReLU()),
            Conv(c_, c2, 1, 1))
    
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
