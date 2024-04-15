import os
import cv2 
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
import urllib.request
import torch
import torch.nn as nn
import tqdm
import numpy as np
import cv2
import cv2
import imageio
import torch
import numpy as np
from tqdm.notebook import tqdm as tqdm
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
import re
def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor * 256
    tensor[tensor > 255] = 255
    tensor[tensor < 0] = 0
    tensor = tensor.type(torch.uint8).permute(1, 2, 0).cpu().numpy()

    return tensor

def get_image(url):
    image_url = url
    img = imageio.imread(image_url)[..., :3] / 255.
    c = [img.shape[0] // 2, img.shape[1] // 2]
    r = 256
    img = img[c[0] - r:c[0] + r, c[1] - r:c[1] + r]

    return img
   
    
class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=32):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


def get_model():

    model = nn.Sequential(
            nn.Conv2d(
                256,
                256,
                kernel_size=1,
                padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(
                256,
                256,
                kernel_size=1,
                padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(
                256,
                256,
                kernel_size=1,
                padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        
            nn.Conv2d(
                256,
                256,
                kernel_size=1,
                padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(
                256,
                3,
                kernel_size=1,
                padding=0),
            nn.Sigmoid(),
        ).to(device)
    return model
def gen_image_Gauss_0(train_dir,output):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index=0;
    image_temp=[]

    for image in os.listdir(train_dir):
        # if os.path.exists(output+ "\\"+ re.sub(".jpg","",image)  ) == False:
            # os.mkdir(output+ "\\"+ re.sub(".jpg","",image) )
        image_temp.append(image)

    target = torch.tensor(get_image(train_dir + "\\"+os.fsdecode(image_temp[0]))).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    # print(target.shape)

    coords = np.linspace(0, 1, target.shape[2], endpoint=False)
    xy_grid = np.stack(np.meshgrid(coords, coords), -1)
    xy_grid1 = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(device)
    x = GaussianFourierFeatureTransform(2, 128, scale = 10)(xy_grid1)
    model = get_model()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)

    for image in image_temp:
        target = torch.tensor(get_image(train_dir + "\\"+os.fsdecode(image))).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        
        for epoch in tqdm(range(1,201)):
            optimizer.zero_grad()
            generated = model(x)
            loss = torch.nn.functional.l1_loss(target, generated)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print('Epoch %d, loss = %.03f' % (epoch, float(loss)))
                plt.imsave(output + "\\" +  re.sub(".jpg","",image)+"_"+str(epoch)+".jpg"  , tensor_to_numpy(generated[0]))

def gen_image_Gauss_2(train_dir,output):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index=0;
    image_temp=[]

    for image in os.listdir(train_dir):
        # if os.path.exists(output+ "\\"+ re.sub(".jpg","",image)  ) == False:
            # os.mkdir(output+ "\\"+ re.sub(".jpg","",image) )
        image_temp.append(image)

    target = torch.tensor(get_image(train_dir + "\\"+os.fsdecode(image_temp[0]))).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    # print(target.shape)

    coords = np.linspace(0, 1, target.shape[2], endpoint=False)
    xy_grid = np.stack(np.meshgrid(coords, coords), -1)
    xy_grid1 = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(device)
    x = GaussianFourierFeatureTransform(2, 128, scale = 10)(xy_grid1)
    model = get_model()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)

    for image in image_temp:
        target = torch.tensor(get_image(train_dir + "\\"+os.fsdecode(image))).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        
        for epoch in tqdm(range(1,201)):
            optimizer.zero_grad()
            generated = model(x)
            loss = torch.nn.functional.l1_loss(target, generated)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print('Epoch %d, loss = %.03f' % (epoch, float(loss)))
                plt.imsave(output + "\\" +  re.sub(".jpg","",image)+"_"+str(epoch)+".jpg"  , tensor_to_numpy(generated[0]))

def gen_image_Gauss_4(train_dir,output):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index=0;
    image_temp=[]

    for image in os.listdir(train_dir):
        # if os.path.exists(output+ "\\"+ re.sub(".jpg","",image)  ) == False:
            # os.mkdir(output+ "\\"+ re.sub(".jpg","",image) )
        image_temp.append(image)

    target = torch.tensor(get_image(train_dir + "\\"+os.fsdecode(image_temp[0]))).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    # print(target.shape)

    coords = np.linspace(0, 1, target.shape[2], endpoint=False)
    xy_grid = np.stack(np.meshgrid(coords, coords), -1)
    xy_grid1 = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(device)
    x = GaussianFourierFeatureTransform(2, 128, scale = 10)(xy_grid1)
    model = get_model()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)

    for image in image_temp:
        target = torch.tensor(get_image(train_dir + "\\"+os.fsdecode(image))).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        
        for epoch in tqdm(range(1,201)):
            optimizer.zero_grad()
            generated = model(x)
            loss = torch.nn.functional.l1_loss(target, generated)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print('Epoch %d, loss = %.03f' % (epoch, float(loss)))
                plt.imsave(output + "\\" +  re.sub(".jpg","",image)+"_"+str(epoch)+".jpg"  , tensor_to_numpy(generated[0]))

def gen_image_Gauss_6(train_dir,output):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index=0;
    image_temp=[]

    for image in os.listdir(train_dir):
        # if os.path.exists(output+ "\\"+ re.sub(".jpg","",image)  ) == False:
            # os.mkdir(output+ "\\"+ re.sub(".jpg","",image) )
        image_temp.append(image)

    target = torch.tensor(get_image(train_dir + "\\"+os.fsdecode(image_temp[0]))).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    # print(target.shape)

    coords = np.linspace(0, 1, target.shape[2], endpoint=False)
    xy_grid = np.stack(np.meshgrid(coords, coords), -1)
    xy_grid1 = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(device)
    x = GaussianFourierFeatureTransform(2, 128, scale = 10)(xy_grid1)
    model = get_model()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)

    for image in image_temp:
        target = torch.tensor(get_image(train_dir + "\\"+os.fsdecode(image))).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        
        for epoch in tqdm(range(1,201)):
            optimizer.zero_grad()
            generated = model(x)
            loss = torch.nn.functional.l1_loss(target, generated)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print('Epoch %d, loss = %.03f' % (epoch, float(loss)))
                plt.imsave(output + "\\" +  re.sub(".jpg","",image)+"_"+str(epoch)+".jpg"  , tensor_to_numpy(generated[0]))

def gen_image_Gauss_8(train_dir,output):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index=0;
    image_temp=[]

    for image in os.listdir(train_dir):
        # if os.path.exists(output+ "\\"+ re.sub(".jpg","",image)  ) == False:
            # os.mkdir(output+ "\\"+ re.sub(".jpg","",image) )
        image_temp.append(image)

    target = torch.tensor(get_image(train_dir + "\\"+os.fsdecode(image_temp[0]))).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    # print(target.shape)

    coords = np.linspace(0, 1, target.shape[2], endpoint=False)
    xy_grid = np.stack(np.meshgrid(coords, coords), -1)
    xy_grid1 = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(device)
    x = GaussianFourierFeatureTransform(2, 128, scale = 10)(xy_grid1)
    model = get_model()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)

    for image in image_temp:
        target = torch.tensor(get_image(train_dir + "\\"+os.fsdecode(image))).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        
        for epoch in tqdm(range(1,201)):
            optimizer.zero_grad()
            generated = model(x)
            loss = torch.nn.functional.l1_loss(target, generated)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print('Epoch %d, loss = %.03f' % (epoch, float(loss)))
                plt.imsave(output + "\\" +  re.sub(".jpg","",image)+"_"+str(epoch)+".jpg"  , tensor_to_numpy(generated[0]))


train_dir=
output = 
gen_image_Gauss_0(train_dir, output)

train_dir=
output =
gen_image_Gauss_2(train_dir, output)

train_dir=
output = 
gen_image_Gauss_4(train_dir, output)

train_dir=
output = 
gen_image_Gauss_6(train_dir, output)

train_dir=
output = 
gen_image_Gauss_8(train_dir, output)
