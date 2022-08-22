import os
import torchvision.models as models
from test_torch import test_torch_main
from torch2onnx import torch2onnx_main
from quantization import quantization_main
from test_int8trt import test_int8trt_main

if __name__ == "__main__":
    net = models.resnet101(pretrained=True).to('cpu')
    test_torch_main(net)
    torch2onnx_main(net)
    quantization_main(net)
    test_int8trt_main()
    os.system("du -sh ./model/*")


