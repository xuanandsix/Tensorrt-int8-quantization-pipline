import torch
import torch.nn as nn
import os
from PIL import Image
import csv
import torchvision.models as models
import json 
import numpy as np
import time
from torchvision import transforms

def load_data(data_path, labels):
    images = []
    targets = []
    for folder in os.listdir(data_path):
        full_dir = os.path.join(data_path, folder)
        for filename in os.listdir(full_dir):
            img = Image.open(os.path.join(full_dir,filename)).convert('RGB')
            if img is not None:
                images.append(img)
                targets.append(labels.index(folder))
    return images, targets

def transform_data(data):
  for index, img in enumerate(data):
      preprocess = transforms.Compose([
      transforms.Resize(224),
      #transforms.Resize(256),
      #transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
      input_tensor = preprocess(img)
      input_batch = input_tensor.unsqueeze(0)
      data[index] = input_batch
  return data


def evaluate(model, data, target):
  model.eval()
  total_time, correct = 0, 0
  with torch.no_grad():
    for img, target in zip(data, target):
      start = time.time()
      output = model(img)
      end = time.time()
      delta = end - start
      total_time += delta
      pred_idx = np.argmax(output[0])
      if target == pred_idx:
        correct += 1
  inference_time = total_time/len(data)
  accuracy = (correct/len(data))*100
  return inference_time, accuracy

def test_torch_main(net):
  print("Run Test Pytorch ....")
  torch.save(net.state_dict(), "./model/model.pth")
  data_path = "./imagenet_1k/"
  f_labels = open(data_path + "labels.txt",'r')
  labels = [x.split(',')[0] for x in f_labels.readlines()]
  float_model = net
  valdir = os.path.join(data_path, 'val')
  test_data, test_labels = load_data(valdir, labels)

  test_data = transform_data(test_data)
  inference_time, accuracy = evaluate(float_model, test_data, test_labels)
  print("Pytorch Inference Time: ", inference_time)
  print("Pytorch Accuracy: ", accuracy, '%')

if __name__ == '__main__':
  net = models.resnet101(pretrained=True).to('cpu')
  test_torch_main(net)
