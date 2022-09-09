import numpy as np
import torch
import torch.nn as nn
import util_trt
import glob,os,cv2
import argparse


def preprocess(image):
    swap=(2, 0, 1)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    input_size = [640, 640]
    if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img

class DataLoader:
    def __init__(self, BATCH, batch_size):
        self.index = 0
        self.length = BATCH
        self.batch_size = batch_size
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(calibration_path, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, '{} must contains more than '.format(calibration_path) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size,3,height,width), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess(img)
                self.calibration_data[i] = img

            self.index += 1

            # example only
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_model_path', type=str, default=None, help='onnx model path')
    parser.add_argument('--calibration_path', type=str, default="./calibration/", help='calibration image path')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')

    args = parser.parse_args()
    batch_size = args.batch_size
    
    calibration_path = args.calibration_path
    onnx_model_path = args.onnx_model_path
    BATCH = 100
    height = 640
    width = 640
    # onnx2trt
    fp16_mode = False
    int8_mode = True 
    print('*** onnx to tensorrt begin ***')
    # calibration
    calibration_stream = DataLoader(BATCH, batch_size)

    engine_model_path = onnx_model_path.replace("onnx", "trt")
    calibration_table = 'calibration.cache'
    # fixed_engine,校准产生校准表
    engine_fixed = util_trt.get_engine(batch_size, onnx_model_path, engine_model_path, fp16_mode=fp16_mode, 
        int8_mode=int8_mode, calibration_stream=calibration_stream, calibration_table_path=calibration_table, save_engine=True)
    assert engine_fixed, 'Broken engine_fixed'
    print('*** onnx to tensorrt completed ***\n')

