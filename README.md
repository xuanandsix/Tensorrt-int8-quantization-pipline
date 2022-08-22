# Tensorrt-int8-quantization-pipline
a simple pipline of int8 quantization based on tensorrt.  

## Example for classification

#### 1、Choose a model and prepare a calibration dataset，like resnet101 training from imagenet1k.
```
wget https://hanlab.mit.edu/files/OnceForAll/ofa_cvpr_tutorial/imagenet_1k.zip
unzip 'imagenet_1k.zip'
mkdir model
```
#### 2、eval the float32 model performance.
```
python test_torch.py
```
#### 3、convert to onnx model.
```
python torch2onnx.py
```
#### 4、 quantization int8 trt model.
```
python quantization.py
```
#### 5、eval the int8 model performance.
```
python test_int8trt.py
```

or run a pipline including the above steps.
```
python tensorrt_PTA_classification_pipline.py
```

<img src="https://github.com/xuanandsix/Tensorrt-int8-quantization-pipline/raw/main/classification/shot.jpg" width="400px" height="380px">

| model | accuracy | time | size |
| :-: |:-:| :-:|:-:|
| float32(pth)|0.759 | 0.0799 |171M|
| int8(trt)|0.738 | 0.0013 | 44M |

#### Note
You can replace resnet101 with your network. If your dataset structure is different, you need to modify some code about dataset.
```
# test_torch.py torch2onnx.py quantization.py
if __name__ == "__main__":
    net = models.resnet101(pretrained=True).to('cpu')
```
or
```
# tensorrt_PTA_classification_pipline.py
if __name__ == "__main__":
    net = models.resnet101(pretrained=True).to('cpu')
```


### TO DO
- [ ] example for detection.
- [ ] example for segmention.
