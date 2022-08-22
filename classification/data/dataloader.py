import pickle
import os
import numpy as np 
import torchvision.transforms as transforms
import torch
import cv2

class Imagenet1k():
    def __init__(self, root):
        self.data = []
        self.targets = []
        f_labels = open("./imagenet_1k/labels.txt",'r')
        labels = [x.split(',')[0] for x in f_labels.readlines()]
        f_txt = open("./data.txt",'r')
        files = f_txt.readlines()
        path = './imagenet_1k/train/'
        for f in files:
            f = f.strip('\n')
            cla, name = f.split("/")
            img = cv2.imread(path + f)
            img = cv2.resize(img, (224, 224))
            self.data.append(img)
            self.targets.extend([labels.index(cla)])
        self.data = np.vstack(self.data).reshape(-1, 3, 224, 224)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.idx = 0

    def __len__(self):
        return len(self.data)

    def get_batch_images(self, current_idx, batch_size):
        
        transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        imgs = np.zeros((batch_size, 3, 224, 224), dtype=np.float32)
        for i in range(batch_size):
            img, _ = self.get_one_image(current_idx + i)

            imgs[i,:,:,:] = transform(img).numpy()


        # import pdb;pdb.set_trace()
        return imgs


    def get_one_image(self, idx=None):

        if idx:
            self.idx = idx 
        
        if self.idx >= len(self):
            self.idx = 0

        img, target = self.data[self.idx], self.targets[self.idx]

        # self.idx += 1

        return img, target

    def get_one_image_torch(self, idx=None):

        img, target = self.get_one_image(idx)

        transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = transform(img)
        # import pdb;pdb.set_trace()
        return img.unsqueeze(0), target
    
    
    def get_one_image_trt(self, pagelocked_buffer, idx=None):

        img, target = self.get_one_image(idx)

        transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = transform(img).view(-1).numpy()

        np.copyto(pagelocked_buffer, img)



