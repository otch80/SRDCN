import cv2
import numpy as np
from glob import glob
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self): # 데이터셋 전처리 (이미지 패치)
        inputImgFolder = "./dataset/T91_ILR"
        labelImgFolder = "./dataset/T91_HR"
        patchSize = 32

        inputImgPaths = glob("%s/*.png" % (inputImgFolder)) # glob 를 통해 png 확장자 파일 가져오기
        labelImgPaths = glob("%s/*.png" % (labelImgFolder))
        inputImgPaths.sort()
        labelImgPaths.sort()

        self.inputPatchs = []
        self.labelPatchs = []

        for idx in range(len(inputImgPaths)):
            # ILR, HR 이미지 각각 읽기, 정규화
            inputImg = np.array(cv2.imread(inputImgPaths[idx]), dtype=np.float32) / 255.
            labelImg = np.array(cv2.imread(labelImgPaths[idx]), dtype=np.float32) / 255.

            # 이미지 차원 변경 (H x W x C -> C x H x W)
            inputImg = np.transpose(inputImg, [2, 0, 1])
            labelImg = np.transpose(labelImg, [2, 0, 1])

            # 한개 이미지를 여러개의 패치로 변경
            self.frameToPatchs(inputImg=inputImg, labelImg=labelImg, patchSize=patchSize)

    def __len__(self): # 데이터셋 개수 반환
        return len(self.inputPatchs)

    def __getitem__(self, idx): # idx 번째 데이터 반환
        return self.inputPatchs[idx], self.labelPatchs[idx]

    def frameToPatchs(self, inputImg=None, labelImg=None, patchSize=32):
        channel, height, width = labelImg.shape

        numPatchY = height // patchSize
        numPatchX = width // patchSize

        for yIdx in range(numPatchY):
            for xIdx in range(numPatchX):
                xStartPos = xIdx * patchSize
                xFianlPos = (xIdx * patchSize) + patchSize
                yStartPos = yIdx * patchSize
                yFianlPos = (yIdx * patchSize) + patchSize

                self.inputPatchs.append(inputImg[:, yStartPos:yFianlPos, xStartPos:xFianlPos])
                self.labelPatchs.append(labelImg[:, yStartPos:yFianlPos, xStartPos:xFianlPos])


class TestDataset(Dataset):
    def __init__(self):
        inputImgFolder = "./dataset/Set5_ILR"
        labelImgFolder = "./dataset/Set5_HR"

        inputImgPaths = glob("%s\\*.bmp" % (inputImgFolder))
        labelImgPaths = glob("%s\\*.bmp" % (labelImgFolder))
        inputImgPaths.sort()
        labelImgPaths.sort()

        self.inputImgs = []
        self.labelImgs = []
        self.imgName = [] # 이미지 이름을 저장하기 위한 리스트 추가 생성

        for idx in range(len(inputImgPaths)):
            inputImg = np.array(cv2.imread(inputImgPaths[idx]), dtype=np.float32) / 255.
            labelImg = np.array(cv2.imread(labelImgPaths[idx]), dtype=np.float32) / 255.

            inputImg = np.transpose(inputImg, [2, 0, 1])
            labelImg = np.transpose(labelImg, [2, 0, 1])

            # 한 개 이미지를 그대로 리스트에 저장
            self.inputImgs.append(inputImg)
            self.labelImgs.append(labelImg)
            self.imgName.append(inputImgPaths[idx].split("/")[-1])

    def __len__(self):
        return len(self.inputImgs)

    def __getitem__(self, idx):
        return self.inputImgs[idx], self.labelImgs[idx], self.imgName[idx] # 3번째 return 값으로 이미지 이름 추가