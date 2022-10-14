import cv2
import os
import argparse
import random
import shutil
def sample(traing_data_path, count, calibration_path):
    files = os.listdir(traing_data_path)
    random.shuffle(files)

    for f in files[:count]:
        shutil.copyfile(traing_data_path + f, calibration_path + f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traing_data_path', type=str, default=None, help='traing data path')
    parser.add_argument('--count', type=int, default=2000, help='calibration data count')
    parser.add_argument('--calibration_path', type=str, default="./calibration/", help='save calibration image path')
    args = parser.parse_args()
    sample(args.traing_data_path, args.count, args.calibration_path)


