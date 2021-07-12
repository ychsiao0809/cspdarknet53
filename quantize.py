import sys
sys.path.append('./darknet53')
sys.path.append('./cspdarknet53')

from darknet53 import DarkNet53
from layers import Conv2dBatchLeaky
from csdarknet53 import CsDarkNet53
from tqdm import trange

import time
import torch

def quantize_model(model, backend):
    _dummy_input_data = torch.rand(1, 3, 384, 640)
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
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

class QuantizableDarkNet(DarkNet53):
    def __init__(self, *args, **kwargs):
        super(QuantizableDarkNet, self).__init__(*args, **kwargs)
        
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = super(QuantizableDarkNet, self).forward(x)
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
        pass 

def main():
#    qdn = QuantizableDarkNet(2)
    qdn = QuantizableCSPDarkNet(2, 'relu6')
    qdn.eval()
    quantize_model(qdn, 'qnnpack')
    start = time.time()
    img = torch.randn(1,3,384,640)
    with torch.no_grad():
        with trange(20) as t:
            for _ in t:
                qdn.forward(img)
    end = time.time()
    print(end - start)

if __name__ == '__main__':
   main()
