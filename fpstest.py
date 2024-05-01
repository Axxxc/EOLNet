import argparse
import logging
import time
import torch

from models.yolo import attempt_load
from utils.torch_utils import select_device


def test():
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    
    device = select_device(opt.device)
    model = attempt_load(opt.weight, map_location=device)
    
    batch_size = opt.batch_size
    img = torch.rand([batch_size, 3, 640, 640], device=device, dtype=torch.float32)
    
    if opt.accelerate:
        assert device.type == 'cpu', 'hardware acceleration is only for CPU'
        model.accelerate()
    
    half = (device.type != 'cpu')
    if half:
        model.half()
        img = img.half()
    
    t = 0.
    for i in range(1000):
        # Run model
        ts = time.time()
        with torch.no_grad():
            model(img)
        t += time.time() - ts
        print('{} / {}'.format(i+1, 1000), end = "\r")
    
    return t * 1E3 / 1000 / batch_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='', help='*.pt path')
    parser.add_argument('--batch-size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--accelerate', action='store_true', help='activate hardware acceleration')
    opt = parser.parse_args()
    
    print('\nSpeed: %.2f ms per image' % test())
