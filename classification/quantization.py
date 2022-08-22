import torch
import datetime
import time
from tqdm import tqdm

import random
from PIL import Image
import numpy as np
import sys
import argparse
import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

#import tensorrt as trt
import torchvision.models as models

from trt.utils import *
from trt import common
from trt.calibrator import ResNetEntropyCalibrator

# from models import resnet18
from collections import OrderedDict

from data.dataloader import Imagenet1k

class ModelData(object):
    INPUT_SHAPE = (3, 224, 224)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32


def quantization_main(model):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    onnx_model_file = './model/model.onnx'
    inference_times = 100
    dataset_test = Imagenet1k('./datasets')
    # ==> pytorch test
    state_dict = torch.load('./model/model.pth')
    model.load_state_dict(state_dict, strict=True)
    model_torch = model
    input_torch, _ = dataset_test.get_one_image_torch(idx=0)

    t_begin = time.time()
    model_torch.eval()
    model_torch.cuda()

    with torch.no_grad():
        for i in tqdm(range(inference_times)):
            outputs_torch = model_torch(input_torch.cuda())
    t_end = datetime.datetime.now()
    
    t_end = time.time()
    torch_time = (t_end - t_begin)/inference_times
    
    # ==> trt test
    with build_engine_onnx(TRT_LOGGER, onnx_model_file) as engine:

        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:

            # Load a normalized test case into the host input page-locked buffer.
            dataset_test.get_one_image_trt(inputs[0].host, idx=0)
            
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            
            t_begin = time.time()
            for i in tqdm(range(inference_times)):
                trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            t_end = time.time()
            trt_time = (t_end - t_begin)/inference_times

    # ==> trt int8 quantization test
    calibration_cache = './model/modelInt8.engine'
    training_data = './datasets'
    # get the calibrator for int8 post-quantization
    calib = ResNetEntropyCalibrator(training_data=training_data, cache_file=calibration_cache)

    with build_engine_onnx_int8(TRT_LOGGER, onnx_model_file, calib) as engine:

        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:

            # Load a normalized test case into the host input page-locked buffer.
            dataset_test.get_one_image_trt(inputs[0].host, idx=0)
            
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            t_begin = time.time()
            for i in tqdm(range(inference_times)):
                trt_int8_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            t_end = time.time()
            trt_int8_time = (t_end - t_begin)/inference_times
        with open('./model/modelInt8.engine', "wb") as f:
            f.write(engine.serialize())

    print('==> Torch time: {:.5f} ms'.format(torch_time))
    print('==> TRT time: {:.5f} ms'.format(trt_time))
    print('==> TRT INT8 time: {:.5f} ms'.format(trt_int8_time))

if __name__ =='__main__':
    model = models.resnet101(pretrained=True).to('cpu')
    quantization_main(model)
