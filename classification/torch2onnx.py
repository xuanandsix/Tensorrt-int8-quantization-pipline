# -*- coding: utf-8 -*-
#import cv2
import numpy as np
import time
import torch
import pdb
from collections import OrderedDict
import sys
import onnxruntime
import torchvision.models as models

def torch2onnx_main(net):
    #net.load_state_dict(torch.load("./model.pth", map_location=torch.device('cpu')))
    input = torch.randn(1, 3, 224, 224, device='cpu')
    torch.onnx.export(net, input, './model/model.onnx',
                    export_params=True, opset_version=11, do_constant_folding=True,
                    input_names = ['input'])
    print("Convert onnx done!")
if __name__ == '__main__':
  net = models.resnet101(pretrained=True).to('cpu')
  torch2onnx_main(net)
