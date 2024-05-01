import argparse
import logging
import time
import torch

from models.yolo import attempt_load
from models.common import *
from utils.torch_utils import select_device, fuse_conv_and_bn


def quantize():
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    
    model = attempt_load(opt.weight)
    
    for m in model.model.modules():
        if type(m) is Conv and hasattr(m, 'bn'):
            m.conv = fuse_conv_and_bn(m.conv, m.bn) 
            delattr(m, 'bn')
    
    for m in model.model.modules():
        if type(m) is Conv:
            if isinstance(m.act, nn.Identity):
                m = QuantConv2d(m.conv, m.bn)
            else:
                m = nn.Sequential(
                    QuantConv2d(),
                    m.act()
                )
    
    return None


def weight_quantize_mixed(model):
    for m in model.model.modules():
        if type(m) is QuantConv2d:
            wmax = max(-torch.min(m.weight), torch.max(m.weight)).to(torch.float32)
            
            m.wscale = 6 - math.floor(torch.log2(wmax)) if wmax >= 2 ** -126 else 0
            qweight = torch.round(m.weight * (2 ** m.wscale))
            qweight = torch.where(qweight <= 127., qweight, torch.tensor(127., dtype=torch.float32, device=qweight.device).expand_as(qweight))
            qweight = torch.where(qweight >= -128., qweight, torch.tensor(-128., dtype=torch.float32, device=qweight.device).expand_as(qweight))
            m.qweight = qweight.to(torch.int8)
    
    return model


def weight_quantize_mixed(model):
    for m in model.model.modules():
        if type(m) is QuantConv2d:
            wmax = max(-torch.min(m.weight), torch.max(m.weight)).to(torch.float32)
                
            m.wscale = 14 - math.floor(torch.log2(wmax)) if wmax >= 2 ** -126 else 0
            qweight = torch.round(m.weight * (2 ** m.wscale))
            qweight = torch.where(qweight <= 32767., qweight, torch.tensor(32767., dtype=torch.float32, device=qweight.device).expand_as(qweight))
            qweight = torch.where(qweight >= -32768., qweight, torch.tensor(-32768., dtype=torch.float32, device=qweight.device).expand_as(qweight))
            m.qweight = qweight.to(torch.int16)
    
    return model


def qat(self):
        for _, v in self.named_parameters():
            v.requires_grad = False
        
        self.fuse()
        
        for m in self.model.modules():
            if type(m) is Conv and type(m.conv) is nn.Conv2d:
                m.conv = QuantConv2d(m.conv)
        return self
    
    def set_max_record(self, record=False):
        for m in self.model.modules():
            if type(m) is Conv:
                m.record = record
        return self
    
    def clear_max_record(self):
        for m in self.model.modules():
            if type(m) is Conv:
                m.xmax[0] = 0
                m.omax[0] = 0
            elif type(m) is QuantConv2d:
                m.xscale = 15
        return self
    
    # post training quantization
def quantization(self):
        print('quantizing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and type(m.conv) is nn.Conv2d:
                xscale = math.floor(15 - torch.log2(m.xmax).item())
                xscale = min(xscale, 30)
                m.conv = QuantConv2d(m.conv, xscale)
                m.conv.forward = m.conv.dforward
        
        self.weight_quantize()
        
        #### 16bit fixed normal quantization ####
        # for m in self.model.modules():
            # if type(m) is Conv:
                # oscale = 15 - torch.log2(m.omax).item()
                # if m.conv.xscale + m.conv.wscale >= oscale:
                    # m.conv.xscale = math.ceil(m.conv.xscale * oscale // (m.conv.xscale + m.conv.wscale))
                    # m.conv.wscale = math.floor(oscale - m.conv.xscale) if oscale % 1 else int(oscale - m.conv.xscale - 1)
                    # qweight = torch.round(m.conv.weight * (2 ** m.conv.wscale))
                    # qweight = torch.where(qweight <= 32767., qweight, torch.tensor(32767., dtype=torch.float32, device=qweight.device).expand_as(qweight))
                    # qweight = torch.where(qweight >= -32768., qweight, torch.tensor(-32768., dtype=torch.float32, device=qweight.device).expand_as(qweight))
                    # m.conv.qweight = qweight.to(torch.int16)
        
        return self
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='', help='*.pt path')
    parser.add_argument('--method', type=int, default=1, help='size of each image batch')
    opt = parser.parse_args()
    
    quantize()
