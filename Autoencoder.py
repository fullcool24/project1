# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:21:22 2019

@author: suchismitasa
"""

import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

import numpy as np


# Data Preprocessing

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])
trainset = tv.datasets.MNIST(root='./data',  train=True,download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=4)
testset = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# Defining Model

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(32,32,kernel_size=3),
            nn.ReLU(True),

			nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(True),
            nn.Conv2d(32,32,kernel_size=3),
            nn.ReLU(True))

        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(32,32,kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,32,kernel_size=3),
            nn.ReLU(True),
			nn.ConvTranspose2d(32,32,kernel_size=3),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,6,kernel_size=3),
            nn.ReLU(True),
            nn.Sigmoid())

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def tie_weights(src, trg):
	assert type(src) == type(trg)
	trg.weight = src.weight
	trg.bias = src.bias

class CenterCrop(nn.Module):
	"""Center-crop if observation is not already cropped"""
	def __init__(self, size):
		super().__init__()
		assert size == 84
		self.size = size

	def forward(self, x):
		assert x.ndim == 4, 'input must be a 4D tensor'
		if x.size(2) == self.size and x.size(3) == self.size:
			return x
		elif x.size(-1) == 100:
			return x[:, :, 8:-8, 8:-8]
		else:
			return ValueError('unexepcted input size')

class NormalizeImg(nn.Module):
	"""Normalize observation"""
	def forward(self, x):
		return x/255.

class PixelEncoder(nn.Module):
	"""Convolutional encoder of pixel observations"""
	def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, num_shared_layers=4):
		super().__init__()
		assert len(obs_shape) == 3

		self.feature_dim = feature_dim
		self.num_layers = num_layers
		self.num_shared_layers = num_shared_layers

		self.preprocess = nn.Sequential(
			CenterCrop(size=84), NormalizeImg()
		)

		self.convs = nn.ModuleList(
			[nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
		)
		for i in range(num_layers - 1):
			self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

		out_dim = OUT_DIM[num_layers]
		self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
		self.ln = nn.LayerNorm(self.feature_dim)

	def forward_conv(self, obs, detach=False):
		obs = self.preprocess(obs)
		conv = torch.relu(self.convs[0](obs))

		for i in range(1, self.num_layers):
			conv = torch.relu(self.convs[i](conv))
			if i == self.num_shared_layers-1 and detach:
				conv = conv.detach()

		h = conv.view(conv.size(0), -1)
		return h

	def forward(self, obs, detach=False):
		h = self.forward_conv(obs, detach)
		h_fc = self.fc(h)
		h_norm = self.ln(h_fc)
		out = torch.tanh(h_norm)

		return out

	def copy_conv_weights_from(self, source, n=None):
		"""Tie n first convolutional layers"""
		if n is None:
			n = self.num_layers
		for i in range(n):
			tie_weights(src=source.convs[i], trg=self.convs[i])

class PixelDecoder(nn.Module):
	"""Convolutional decoder of pixel observations"""
	def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, num_shared_layers=4):
		super().__init__()
		# assert len(obs_shape) == 3

		self.feature_dim = feature_dim
		self.num_layers = num_layers
		self.num_shared_layers = num_shared_layers

		self.convs = nn.ModuleList(
			[nn.ConvTranspose2d(obs_shape[0], num_filters, 3, stride=2)]
		)
		for i in range(num_layers - 1):
			self.convs.append(nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1))

		out_dim = OUT_DIM[num_layers]
		self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
		self.ln = nn.LayerNorm(self.feature_dim)

	def forward_conv(self, obs, detach=False):
		conv = torch.relu(self.convs[0](obs))

		for i in range(1, self.num_layers):
			conv = torch.relu(self.convs[i](conv))
			if i == self.num_shared_layers-1 and detach:
				conv = conv.detach()

		h = conv.view(conv.size(0), -1)
		return h

	def forward(self, obs, detach=False):
		h = self.forward_conv(obs, detach)
		h_fc = self.fc(h)
		h_norm = self.ln(h_fc)
		out = torch.tanh(h_norm)

		return out

	def copy_conv_weights_from(self, source, n=None):
		"""Tie n first convolutional layers"""
		if n is None:
			n = self.num_layers
		for i in range(n):
			tie_weights(src=source.convs[i], trg=self.convs[i])

# Defining Parameters

OUT_DIM = {2: 39, 4: 35, 6: 31, 8: 27, 10: 23, 11: 21, 12: 19}

def make_encoder(
	obs_shape, feature_dim, num_layers, num_filters, num_shared_layers
):
	assert num_layers in OUT_DIM.keys(), 'invalid number of layers'
	if num_shared_layers == -1 or num_shared_layers == None:
		num_shared_layers = num_layers
	assert num_shared_layers <= num_layers and num_shared_layers > 0, \
		f'invalid number of shared layers, received {num_shared_layers} layers'
	return PixelEncoder(
		obs_shape, feature_dim, num_layers, num_filters, num_shared_layers
	)

num_epochs = 5
batch_size = 128
obs_shape = (3*2, 84, 84)
hidden_dim=256
encoder_feature_dim=50
num_layers=4
num_shared_layers=4
num_filters=32
encoder = make_encoder(
            obs_shape, encoder_feature_dim, num_layers,
            num_filters, num_shared_layers
        )
print(encoder)

model = Autoencoder().cpu()
print(model)

dummy = np.ones((1,)+obs_shape,dtype=np.float32)
dummy = torch.tensor(dummy)
from_enc = encoder(dummy)
# model(dummy)

def make_decoder(
	obs_shape, feature_dim, num_layers, num_filters, num_shared_layers
):
	assert num_layers in OUT_DIM.keys(), 'invalid number of layers'
	if num_shared_layers == -1 or num_shared_layers == None:
		num_shared_layers = num_layers
	assert num_shared_layers <= num_layers and num_shared_layers > 0, \
		f'invalid number of shared layers, received {num_shared_layers} layers'
	return PixelDecoder(
		obs_shape, feature_dim, num_layers, num_filters, num_shared_layers
	)

num_epochs = 5
batch_size = 128
obs_shape = (1,50)
hidden_dim=256
decoder_feature_dim=50
num_layers=4
num_shared_layers=4
num_filters=32

# print(from_enc.shape)
# quit()

decoder = make_decoder(obs_shape, decoder_feature_dim, num_layers,
            num_filters, num_shared_layers)

# print(decoder)

model = Autoencoder().cpu()
# print(model)

# dummy = np.ones((1,)+obs_shape,dtype=np.float32)
# dummy = torch.tensor(dummy)
decoder(from_enc)

# model(dummy)

# distance = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)

# for epoch in range(num_epochs):
#      for data in dataloader:
#          img, _ = data
#          img = Variable(img).cpu()
#          # ===================forward=====================
#          output = model(img)
#          loss = distance(output, img)
#          # ===================backward====================
#          optimizer.zero_grad()
#          loss.backward()
#          optimizer.step()
#      # ===================log========================
#      print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.item()))
