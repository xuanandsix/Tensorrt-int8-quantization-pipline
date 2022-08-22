# Tensorrt-int8-quantization-pipline
a simple pipline of int8 quantization based on tensorrt.  

## Example for classification

#### 1、Choose a model and prepare a calibration dataset，like resnet101 training from imagenet1k.
```
wget 'https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip'
unzip 'imagenet_1k.zip'
```
#### 2、eval the float32 model performance.
```
python test_torch.py
```
#### 2、convert to onnx model.
```
python torch2onnx.py
```
#### 3、 quantization int8 trt model.
```
python quantization.py
```
#### 4、eval the int8 model performance.
```
python test_int8trt.py
```

or run a pipline including the above steps.
```
python tensorrt_PTA_classification_pipline.py
```

### TO DO
- [ ] example for detection.
- [ ] example for segmention.
