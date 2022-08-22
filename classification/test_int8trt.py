#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time


class ClassificationDemo:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
        f_labels = open("./imagenet_1k/labels.txt",'r')
        self.labels = [x.split(',')[0] for x in f_labels.readlines()]

    def forward(self, img, label):
        h, w = img.shape[:2]
        img = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0 
        img[:,:,0] = (img[:,:,0]-0.485)/0.229
        img[:,:,1] = (img[:,:,1]-0.456)/0.224
        img[:,:,2] = (img[:,:,2]-0.406)/0.225
        img = np.transpose(np.float32(img[:,:,:,np.newaxis]), (3,2,0,1))
        img = np.ascontiguousarray(img)
        cuda.memcpy_htod(self.inputs[0]['allocation'], img)
        start = time.time()
        self.context.execute_v2(self.allocations)
        end = time.time()
        delta = end - start
        outputs = []
        for out in self.outputs:
            output = np.zeros(out['shape'],out['dtype'])
            cuda.memcpy_dtoh(output, out['allocation'])
            outputs.append(output)
        if self.labels.index(label) == np.argmax(output[0]):
            return 1, delta
        else:
            return 0, delta


def test_int8trt_main():
    print("Run Test Tensorrt ....")
    cla = ClassificationDemo('./model/modelInt8.engine')
    path = './imagenet_1k/val/'
    directory = os.listdir(path)
    num = 0 
    rightNum = 0
    totalTime = 0
    for label in directory:
        files = os.listdir(path + label)
        for f in files:
            image = cv2.imread(path + label + '/' + f)
            output, delta = cla.forward(image, label)
            totalTime += delta
            rightNum += output
            num += 1
    inferenceTime = totalTime / num
    accuracy = (rightNum / num) * 100
    print("Tensorrt Int8 Inference Time: ", inferenceTime)
    print("Tensorrt Int8 Accuracy: ", accuracy, '%')

if __name__ == "__main__":
    test_int8trt_main()
