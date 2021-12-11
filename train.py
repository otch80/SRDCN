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


model = SRDCN().to('cuda')

batch_size = 16
learning_rate = 0.01
training_epochs = 15
loss_function = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = TrainDataset()
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4)

test_dataset = TestDataset()
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1)

for epoch in tqdm(range(training_epochs)):
    avg_cost = 0
    total_batch = len(train_dataloader)

    for data in train_dataloader:
        inputImg, labelImg = data

        inputImg = inputImg.to('cuda')
        labelImg = labelImg.to('cuda')

        predImg = model(inputImg)

        loss = loss_function(predImg, labelImg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_cost += loss / total_batch
    print('Epoch: %d Loss = %f' % (epoch + 1, avg_cost))

torch.save(model.state_dict(), "SRDCN.pt")


def calc_psnr(orig, pred):
    return 10. * torch.log10(1. / torch.mean((orig - pred) ** 2))


bicubic_PSNRs = []
srcnn_PSNRs = []

for data in test_dataloader:
    inputImg, labelImg, imgName = data

    inputImg = inputImg.to('cuda')
    labelImg = labelImg.to('cuda')

    with torch.no_grad():
        predImg = model(inputImg).clamp(0.0, 1.0)
    bicubic_PSNRs.append(calc_psnr(labelImg, inputImg))
    srcnn_PSNRs.append(calc_psnr(labelImg, predImg))

    predImg = np.array(predImg.cpu() * 255, dtype=np.uint8)
    predImg = np.transpose(predImg[0, :, :, :], [1, 2, 0])

    cv2.imwrite("./predict/" + imgName[0], predImg)

print('Average PSNR (bicubic)\t: %.4fdB' % (sum(bicubic_PSNRs) / len(bicubic_PSNRs)))
print('Average PSNR (Custom)\t: %.4fdB' % (sum(srcnn_PSNRs) / len(srcnn_PSNRs)))