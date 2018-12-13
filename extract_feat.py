
# coding: utf-8

# In[2]:


# write your codes here
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random 
import os
import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import scipy.io


def extract_feat():
    model_ft = models.vgg16(pretrained=True)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    prep = transforms.Compose([ transforms.ToTensor(), normalize ])

    #print(model_ft)
    model_ft.classifier = model_ft.classifier[:3]
    for param in model_ft.parameters():
        param.requires_grad = False
    #print(model_ft)
    samples = os.listdir('UCF101_release/images_class1')
    rows, cols = 256, 340
    crow, ccol = rows/2,cols/2
    for sample in samples:
        imgs = os.listdir('UCF101_release/images_class1/' + sample)
        path = 'UCF101_release/images_class1/' + sample + '/'
        path2 = 'UCF101_release/cls1_feat/' + sample +'.mat'
        #print(path2)
        feat = []
        for ig in imgs:
            img = cv2.imread(path + ig)
            #print("before",img.shape)
            img = prep(img)
            #print("after", img.shape)
            #print(img.shape)
            img1 = img[:,:224,:224]
            img2 = img[:,rows-224:,:224]
            img3 = img[:,:224,cols-224:]
            img4 = img[:,rows-224:,cols-224:]
            img5 = img[:,crow-112:crow+112,ccol-112:ccol+112]
            inputs = torch.cat((img1.unsqueeze(0),img2.unsqueeze(0),img3.unsqueeze(0),img4.unsqueeze(0),img5.unsqueeze(0)),0)

            feat1 = model_ft(inputs)
            feat1 = torch.mean(feat1,dim = 0)
            feat.append(np.array(feat1))
        feat = np.array(feat)
        print(feat.shape)
        mydict = {'Feature':feat}
        scipy.io.savemat(path2,mydict)
        
extract_feat()

