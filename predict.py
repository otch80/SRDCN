import cv2
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from srdcn import SRDCN



file_name = 'SRDL_GPU.pt'
model = SRDL()
model.load_state_dict(torch.load(file_name))
# model.load_state_dict(torch.load('model_state.pth', map_location='cpu')) # CPU 사용

inputImg_path = "./TEST.jpg"
inputImg = np.array(cv2.imread(inputImg_path), dtype=np.float32) / 255.
inputImg = np.transpose(inputImg, [2, 0, 1])
inputImg = torch.from_numpy(inputImg)
inputImg = inputImg.reshape(1,inputImg.shape[0],inputImg.shape[1],inputImg.shape[2])

with torch.no_grad():
    predImg = model(inputImg).clamp(0.0, 1.0)

predImg = np.array(predImg.cpu() * 255, dtype=np.uint8)
predImg = np.transpose(predImg[0, :, :, :], [1, 2, 0])

cv2.imwrite("./test_pred.jpg", predImg)