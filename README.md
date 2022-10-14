# Tensorrt-int8-quantization-pipline
a simple pipline of int8 quantization based on tensorrt.  

## Example for classification
```
cd classification
```
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
## Example for detection
```
cd detection
```
#### 1、Choose a model and test inference，like YOLOX-s.
```
wget wget https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1/yolox_s.onnx
python demo_onnx.py --model_path yolox_s.onnx --label_name_path coco.label --image_path dog.jpg --output_path output_onnx.jpg
```

#### 2、random sample 2k training images as calibration data, YOLOX-s training from COCO2017.
```
mkdir calibration
python sample.py --traing_data_path your_path/coco/images/train2017/  --count 2000 --calibration_path ./calibration/
```

#### 3、quantization  
```
python3 -m onnxsim yolox_s.onnx yolox_s.onnx
python quantization.py
```

#### 4、test int tensort model 
```
python demo_trt.py --model_path modelInt8.engine --label_name_path coco.label --image_path dog.jpg --output_path output_trt.jpg
```
| model | time | size |
| :-: |:-:| :-:|
| float32(pth)| 0.0064 |35M|
| int8(trt)| 0.0025 | 9.2M |

| float32 onnx | int8 tensorrt|
| :-: |:-:|
|<img src="https://github.com/xuanandsix/Tensorrt-int8-quantization-pipline/raw/main/detection/show_img/output_onnx.jpg" height="60%" width="60%">|<img src="https://github.com/xuanandsix/Tensorrt-int8-quantization-pipline/raw/main/detection/show_img/output_trt.jpg" height="60%" width="60%">|


### TO DO
- [x] example for detection.
- [ ] example for segmention.
