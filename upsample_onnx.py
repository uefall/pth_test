# 什么原因导致了yolo转onnx时在经过upsample layer后的dynamic出现问题

import torch
from torch import nn
import argparse

class Upsample_net(nn.Module):
    def __init__(self):
        super(Upsample_net, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        up = self.up(x)
        out = self.conv3(up)
        return out
    
def parse_agr():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dynamic', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = parse_agr()
    model = Upsample_net()
    input = torch.randn(8, 1, 32, 32)
    output = model(input)
    print(output.shape)
    if opt.dynamic:
        torch.onnx.export(model, input, "upsample_dynamic.onnx",
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input' : {0 : 'batch_size'},
                                        'output' : {0 : 'batch_size'}},
                          verbose=True,
                          opset_version=11)
    else:
        torch.onnx.export(model, input, "upsample.onnx", 
                          input_names=['input'],
                          output_names=['output'],
                          verbose=True,
                          opset_version=11)
