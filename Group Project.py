#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn

import numpy
import torch.optim as optim
import os
import cv2 ##


#gets current device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"The current device is {device}")


folder_data_train = ("./data/train/image/") ## Denote the file place of the train image
folder_mask_train = ('./data/train/mask/') ## Denote the file place of the train mask

folder_data_val = ("./data/val/image/")
folder_mask_val = ('./data/val/mask/')

class CustomDataset(Dataset):
    def __init__(self, image_paths, target_paths):   # initial logic
         self.image_paths = image_paths
         self.target_paths = target_paths
         self.transforms = transforms.Compose(
             [transforms.ToTensor(),
              # transforms.Grayscale(num_output_channels=1),
              transforms.Normalize((0), (0.5),),
              transforms.Resize((96, 96))])


    def __getitem__(self, index):
        image = Image.open(str(self.image_paths) + os.listdir(self.image_paths)[index])#.convert('RGB')
        mask = Image.open(str(self.target_paths) + os.listdir(self.target_paths)[index])#.convert('RGB')
        t_image = self.transforms(image)
        t_mask = self.transforms(mask)
        return t_image, t_mask

    def __len__(self):  # return count of sample we have
        return len(os.listdir(self.image_paths))


train_set = CustomDataset(folder_data_train, folder_mask_train)
valid_set = CustomDataset(folder_data_val, folder_mask_val)

batch_size = 5

#train_set = CustomDataset(folder_mask_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

#test_set = CustomDataset(folder_data_val)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

#classes
classes = ('__background__', 'left ventricle', 'right ventricle', 'myocardium')

# DataLoader = {
#     'train': train_loader,
#     'valid': test_loader,
# }

def imshow(img):
    img = img / 2
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



dataiter = iter(train_loader)
images, labels = dataiter.next()


# show images
imshow(torchvision.utils.make_grid(images))


# dice loss ========================================================================
# PyTorch
def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

# define neural network ===============================================================================
class UNET(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = self.contract_block(in_channels, 32, 7, 3)
        self.conv2 = self.contract_block(32, 64, 3, 1)
        self.conv3 = self.contract_block(64, 128, 3, 1)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64*2, 32, 3, 1)
        self.upconv1 = self.expand_block(32*2, out_channels, 3, 1)

    def __call__(self, x):

        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                 )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
                            torch.nn.BatchNorm2d(out_channels),
                            torch.nn.ReLU(),
                            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                            )
        return expand

# defining net parameters=================================================================================

# moves net to device
net = UNET(1,1).to(device)

#loss fucntion
criterion = dice_loss

# optimiser function
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

mini_batch = 2

#training net========================================================================================

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.to(device)).float().mean()


#goes over data set
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels] and move them to
        # the current device
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics - epoch and loss
        running_loss += loss.item()
        if i % mini_batch == mini_batch-1:  # print every 2000 mini-batches
            print('[%d, %5d] training loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / mini_batch))
            running_loss = 0.0
    running_loss = 0.0

    for i, data in enumerate(valid_loader, 0):
        with torch.no_grad():
            # get the inputs; data is a list of [inputs, labels] and move them to
            # the current device
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)



            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # print statistics - epoch and loss
            running_loss += loss.item()
            if i % mini_batch == mini_batch - 1:  # print every 2000 mini-batches
                print('[%d, %5d] valid loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / mini_batch))
                running_loss = 0.0

print('Finished Training')


# In[ ]:




