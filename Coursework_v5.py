#!/usr/bin/env python
# coding: utf-8

# # Neural Computation 
# ##### By Group 31 

# ## Introduction

# In[70]:


# Importing all the necessary libraries
import torch
import torch.utils.data as data
import cv2
import os
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import time
from datetime import datetime


# Finds the device the data will be computed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"The current device is {device}")

# Create variables which stores the location of the training images and validation images.
train_data_path = './data/train'

val_data_path = './data/val'

test_data_path = './data/test'

torch.set_printoptions(edgeitems=100)

# It visualises the actual MR image and its mask.
def show_image_mask(img, mask, cmap='gray'):
    fig = plt.figure(figsize=(5, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap=cmap)
    plt.axis('off')


# image = cv2.imread(os.path.join(folder_data_train,'image','cmr10.png'), cv2.IMREAD_UNCHANGED) # A random image to find later the size of it.
# mask = cv2.imread(os.path.join(folder_data_train,'mask','cmr10_mask.png'), cv2.IMREAD_UNCHANGED)
# show_image_mask(image, mask, cmap='gray')
# plt.pause(1)
# cv2.imwrite(os.path.join('./','cmr1.png'), mask*85)

# Take an image as input and it prints it.
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# It loads all the images for the training files.
class CustomDataset(data.Dataset):
    def __init__(self, root=''):
        super(CustomDataset, self).__init__()
        self.img_files = glob(os.path.join(root, 'image', '*.png'))
        self.mask_files = []
        for img_path in self.img_files:
            basename = os.path.basename(img_path)
            self.mask_files.append(os.path.join(root, 'mask', basename[:-4] + '_mask.png'))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)


class TestDataset(data.Dataset):
    def __init__(self, root=''):
        super(TestDataset, self).__init__()
        self.img_files = glob(os.path.join(root, 'image', '*.png'))

    def __getitem__(self, index):
        img_path = self.img_files[index]
        data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        return torch.from_numpy(data).float()

    def __len__(self):
        return len(self.img_files)
       


'''
The DoubleConv is a double convolutional layer, this is the building block of the entire 
architecture. It takes an input x which has number of channels in_ch, it convolutes it (with
nn.Conv2d) to number of channels out_ch with 3x3 kernel and padding 1, then it is normalised 
(with nn.BatchNorm2d) and it passed through the ReLU function. Again it convolutes it from 
channels out_ch to same number of channels out_ch with kernel 3 and padding 1, it is 
normalised and it is passed through the ReLU function. It uses padding 1 to keep the size of 
weight and height unchanged. It returns an output with out_ch channels.
'''
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        return x

'''
The In takes an input x and pass it through the DoubleConv class with in_ch channels and
expects output with out_ch channels. It returns the output of DoubleConv. It is used for 
the first input convolution layers.
'''
class In(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(In, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

'''
The Down is a single layer of the encoder. It takes an input x with channels in_ch, it max 
pools it with 2x2 kernel and then it passes it to DoubleConv with in_ch channels and expects 
output with out_ch channels. The max pool halved the height and weight size. It returns an 
output with out_ch channels and halved height and weight.
'''
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x

'''
The Up is a single layer of the decoder that takes an input x1 with in_ch channels and it is up 
sampled with 2x2 kernel. It takes the second input x2 and the upsampled x1 and merge them in 
variable x (with torch.cat). Then x is passed to DoubleConv with in_ch channels and expects 
output with out_ch channels. The result of DoubleConv is returned.
'''
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

'''
The Out is the last convolutional layer. It takes an input x with in_ch channels and it 
returns a convoluted input with out_ch channels. It is convoluted by 1x1 kernel and padding 0.
'''
class Out(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Out, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
#Defining the model
'''
The model Unet2 consists of 5 classes which are the DoubleConv, the In, the Down, the Up 
and the Out. Now it starts the encoder. It takes an input with in_channels channels and 
passes it trough class In, it returns an input with 64 channels. It goes through the class
Down with 64 channels and returns with 128 channels, again it goes through the class Down 
with 128 channels and returns with 256 channels, again it goes through the class Down with 
256 channels and returns with 512 channels and the last part of encoder the input goes 
through the class Down with 512 channels and returns with 512 channels. After that the 
decoder starts, the Up class takes the x5 and x4 which together ha 1024 channels and returns
an output with 256 channels, again the Up class takes the output and x3 which together has
512 channels and returns an output with 128 channels, again the Up class takes the output 
and x2 which together has 256 channels and returns an output with 64 channels and again the 
Up class takes the output and x1 which together has 128 channels and returns an output with 
64 channels. The last part of the decoder is the Out class which takes the last output of
Up class with 64 channels and returns it with "classes" channels.
'''

class Unet2(nn.Module):
    def __init__(self, in_channels, classes):
        super(Unet2, self).__init__()
        self.n_channels = in_channels
        self.n_classes = classes

        self.inc = In(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = Out(64, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


def categorical_dice(mask1, mask2, label_class=1):
    """
    Dice score of a specified class between two volumes of label masks.
    (classes are encoded but by label class number not one-hot )
    Note: stacks of 2D slices are considered volumes.
    Args:
        mask1: N label masks, numpy array shaped (H, W, N)
        mask2: N label masks, numpy array shaped (H, W, N)
        label_class: the class over which to calculate dice scores
    Returns:
        volume_dice
    """
    
    mask1_pos = (mask1 == label_class).astype(np.float32)
    mask2_pos = (mask2 == label_class).astype(np.float32)
    dice = 2 * np.sum(mask1_pos * mask2_pos) / (np.sum(mask1_pos) + np.sum(mask2_pos))
    return dice


# It creates the model that acceps inputs with 1 channel and returns an output with 4 channels.
model = Unet2(1,4).to(device) 


# In[71]:


import torch.nn as nn
# The loss function is the cross entropy loss function.
Loss = nn.CrossEntropyLoss() 
# The optimizer is the stochastic gradient descent with learning rate 0.1 and momentum 0.9.
# optimizer = optim.Adam(model.parameters(),lr = 0.01) https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
optimizer = optim.SGD(model.parameters(), lr=0.1,momentum=0.9)

num_workers = 4
batch_size = 5

# Saves all the images and masks from the training set and validation set in the variables 
# train_set and valid_set respectively.
train_set = CustomDataset(train_data_path)
valid_set = CustomDataset(val_data_path)

# Saves all the images from the test set in the variable test_set.
test_set = TestDataset(test_data_path)

# Making a variable that contains all the images of the training data set and contains the 
# parameters on which the model will be trained.
training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)

# Making a variable that contains all the images of the validating data set and contains the 
# parameters on which the model will be trained.
valid_loader = DataLoader(dataset=valid_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)

# test_set = TestDataset(folder_)
test_loader = DataLoader(dataset=test_set, num_workers=num_workers, batch_size=1, shuffle=False)

# Create two different lists to store the loss values for the training set and validation set.
# So we can then plot them together to see the progress.
train_losses = list()
val_losses = list()

mini_batch =1

for epoch in range(20):  # loop over the dataset multiple times
    # Training our model on training data and putting the model to train mode.
    model.train()
    running_loss = 0.0
    running_dice_loss = 0.0
    # For loop takes one by one all the data of the training set.
    for iteration, sample in enumerate(training_data_loader):
        # Place the image and its corresponding mask to different variables.
        img, mask = sample 
        
        # Get the inputs and move them to the current device
        img = img.to(device)
        mask = mask.to(device)
        # Since the image is black-white, it has channel 1 and python remove the dimension of channel.
        # So we used the unsqueeze in place 1 to recover the number of channel in place 1, BxCxHxW.
        im = img.unsqueeze(1)

        # The optimised gradients set to zero. https://pytorch.org/docs/stable/optim.html
        optimizer.zero_grad()
        
        # Returns the predicted mask for the training image. Forward probacation
        mask_p = model(im)  

        # Calculate the cross entropy loss for the predicted mask and the actual mask.
        loss = Loss(mask_p, mask.long())

        loss.backward()  # Backward probacation
        optimizer.step()
        
        train_losses.append(loss.item()) # Add the loss value to the training loss list.
        #Print the loss of the trained model
        running_loss += loss.item()
        if iteration % mini_batch == mini_batch - 1:  # print every 2000 mini-batches
            print('[%d, %5d] training loss: %.3f' %
                  (epoch + 1, iteration + 1, running_loss / mini_batch))
            running_loss = 0.0
            
    # Validating our model by computing the loss function based on our training set.
    # For loop takes one by one all the data of the validation set.
    for i, data in enumerate(valid_loader, 0):
        # Putting the model to evaluate phase.
        model.eval()
        with torch.no_grad():
            # Place the image and its corresponding mask to different variables.
            image_v, mask_v = data
            
            # Get the inputs and move them to the current device
            image_v = image_v.to(device)
            mask_v = mask_v.to(device)
            
            # Recover the number of channel in place 1
            imagev = image_v.unsqueeze(1)

            # Returns the predicted mask for the validation image
            mask_pr = model(imagev)
            
            val_loss = Loss(mask_pr, mask_v.long())
            
            # Dice_loss = 0
            # out = torch.argmax(mask_pr, 1)
            # We don't include the backgrand in the calculation of the dice score.
            # for i in range(1,4):
                #Dice_loss = Dice_loss+categorical_dice(out.detach().numpy(),labels.detach().numpy(),i)
                #Dice_loss
            # Dice_loss = Dice_loss/3
            
            val_losses.append(val_loss.item()) # Add the loss value to the validation loss list.
            
            #print(categorical_dice(outputs, labels))
            running_loss += loss.item()
            #running_dice_loss += Dice_loss

            if i % mini_batch == mini_batch - 1:  # print every 2000 mini-batches
                print('[%d, %5d] valid loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / mini_batch))
                # print('[%d, %5d] dice loss: %.3f' %
                      # (epoch + 1, i + 1, running_dice_loss/mini_batch))
                running_loss = 0.0
                # running_dice_loss = 0.0


plt.plot(range(len(train_losses)),train_losses,'r',label="Training loss")
plt.plot(range(len(val_losses)),val_losses,'g',label="Validation loss")
plt.xlabel("Training iterations") # Add label on the x-axis
plt.ylabel("Loss") # Add label on the y-axis
# The loss values are between 0 - 0.25, we zoom the plot, so we can compare more easily.
plt.axis([0, 410, 0, 0.25]) # https://stackoverflow.com/questions/11400579/pyplot-zooming-in
plt.legend()
plt.show()

now = datetime.now() # Find the time when the training and validation test finished.
current_time = now.strftime("%H-%M_%d-%m-%y")


# In[72]:


# Saving the model with a desired name.
PATH = './'+ current_time+'.pth'
torch.save(model.state_dict(), PATH)
print("Saved as: "+str(PATH)) # Print the name of the model.
# To know where the training ends and where the test starts.
print('Finished Training') 


# In[73]:


print("Displaying output")
# Loading our saved model in order to use the previous trained model
PATH = './00-17_27-11-21.pth'
model.load_state_dict(torch.load(PATH))


# In[74]:


# Evaluating our model on the testing data and predicting results
with torch.no_grad():
    # Putting the model to evaluate mode
    model.eval()
    # For loop takes one by one all the data of the testing set.
    for i,data in enumerate(test_loader,0):
        # Place the image in a variable.
        image_t = data
        
        # Get the inputs and move them to the current device
        image_t = image_t.to(device)
        
        # Recover the number of channel in place 1
        im = image_t.unsqueeze(1)
        
        # Returns the predicted mask for the testing image.
        mask_pt = model(im)
        
        # Removes the channel dimension. Finds the channel number with the maximum value 
        # and places it in a new torch. That torch is returned.
        out = torch.argmax(mask_pt,1)
        
        # Removes the number of batch.
        mask_n = out.squeeze(0)
        
        torch.set_printoptions(edgeitems=100)
        out = out *0.196
        output = out[0]
        imaget = image_t[0]
        for a in range(1,out.size(0)):
            output = torch.cat((output,out[a]),axis=1)
            imaget = torch.cat((imaget, image_t[a]), axis=1)
        imaget = imaget.cpu()
        output = output.cpu()
        
        # To know which image is showed and the name of the mask.
        print('cmr{}.png,cmr{}_mask.png'.format(i+121,i+121)) 
        
        fig, axes = plt.subplots(nrows=1, ncols=2)
        im = axes[0].imshow(imaget,cmap='gray')
        clim = im.properties()['clim']
        axes[1].imshow(output,cmap='gray')
        
        plt.show()
        
        # It saves the masks as black. If we want to save them as they are printed, we need to 
        # multiply mask_n.numpy() with 85.
        cv2.imwrite(os.path.join('./data/test/mask','cmr{}_mask.png'.format(i+121)), mask_n.numpy())


# In[68]:


def rle_encoding(x):
    '''
    *** Credit to https://www.kaggle.com/rakhlin/fast-run-length-encoding-python ***
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def submission_converter(mask_directory, path_to_save):
    writer = open(os.path.join(path_to_save, "submission.csv"), 'w')
    writer.write('id,encoding\n')

    files = os.listdir(mask_directory)

    for file in files:
        if not file =='.ipynb_checkpoints':
            name = file[:-4]
            mask = cv2.imread(os.path.join(mask_directory, file), cv2.IMREAD_UNCHANGED)

            mask1 = (mask == 1)
            mask2 = (mask == 2)
            mask3 = (mask == 3)

            encoded_mask1 = rle_encoding(mask1)
            encoded_mask1 = ' '.join(str(e) for e in encoded_mask1)
            encoded_mask2 = rle_encoding(mask2)
            encoded_mask2 = ' '.join(str(e) for e in encoded_mask2)
            encoded_mask3 = rle_encoding(mask3)
            encoded_mask3 = ' '.join(str(e) for e in encoded_mask3)

            writer.write(name + '1,' + encoded_mask1 + "\n")
            writer.write(name + '2,' + encoded_mask2 + "\n")
            writer.write(name + '3,' + encoded_mask3 + "\n")

    writer.close()

mask_directory = './data/test/mask' # The path the test masks
path_to_save = './data/Kaggle' # The path we want the submission_converter save tha masks.
submission_converter(mask_directory, path_to_save) # Convert the masks to Kaggle format.

