import sys
sys.path.append('./darknet53')
sys.path.append('./cspdarknet53')

from darknet53 import DarkNet53
from layers import Conv2dBatchLeaky
from csdarknet53 import CsDarkNet53
from cslayers import Conv2dBatchLeaky as CsConv2dBatchLeaky
from tqdm import trange

import time
import torch
import torch.nn as nn
import argparse

def set_backend(backend):
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend

def quantize_model(model, backend):
    _dummy_input_data = torch.rand(1, 3, 384, 640)
    model.eval()
    # Make sure that weight qconfig matches that of the serialized models
    if backend == 'fbgemm':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_per_channel_weight_observer)
    elif backend == 'qnnpack':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_weight_observer)

    model.fuse_model()
    torch.quantization.prepare(model, inplace=True)
    model(_dummy_input_data)
    torch.quantization.convert(model, inplace=True)

    return

class QuantizableDarkNet(nn.Module):
    def __init__(self, num_classes):
        super(QuantizableDarkNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.darknet = DarkNet53(num_classes)
        self.dequant = torch.quantization.DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        x = self.darknet(x)
        x = self.dequant(x)
        return x
    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, Conv2dBatchLeaky):
                m.fuse_model()

class QuantizableCSPDarkNet(CsDarkNet53):
    def __init__(self, *args, **kwargs):
        super(QuantizableCSPDarkNet, self).__init__(*args, **kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = super(QuantizableCSPDarkNet, self).forward(x)
        x = self.dequant(x)
        return x
        
    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, CsConv2dBatchLeaky):
                m.fuse_model()

def main(args):
    img = torch.randn(1,3,384,640)
    set_backend(args.backend)

    if args.load:
        qdn = torch.jit.load(args.load)
    else:
        if args.model is 'darknet':
            qdn = QuantizableDarkNet(2)
        elif args.model is 'cspdarknet':
            qdn = QuantizableCSPDarkNet(2, 'relu6')
        else:
            print("Selected model not available")
            exit(1)
        quantize_model(qdn, args.backend)
        qdn.eval()

    if args.save:
        torch.jit.save(torch.jit.script(qdn), args.save)

    start = time.time()
    img = torch.randn(1,3,384,640)
    with torch.no_grad():
        with trange(args.inference) as t:
            for _ in t:
                qdn.forward(img)
    end = time.time()
    print(end - start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", default=None, help="Save model", type=str)
    parser.add_argument("-l", "--load", default=None, help="Load model from checkpoint", type=str)
    parser.add_argument("-i", "--inference", default=20, help="Number of inference", type=int)
    parser.add_argument("-t", "--model", default="darknet", help="Choose model type", type=str)
    parser.add_argument("-b", "--backend", default="qnnpack", help="Select backend for PyTorch", type=str)
    args = parser.parse_args()
    main(args)
