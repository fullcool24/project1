import torch
import torch.nn as nn

import numpy as np

from encoder import OUT_DIM


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


class PixelDecoder(nn.Module):
	"""Convolutional decoder of pixel observations"""
	def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, num_shared_layers=4):
		super().__init__()
		# assert len(obs_shape) == 3

		self.feature_dim = feature_dim
		self.num_layers = num_layers
		self.num_filters = num_filters
		self.num_shared_layers = num_shared_layers
		self.out_dim = OUT_DIM[num_layers]

		self.fc = nn.Linear(
			feature_dim, num_filters * self.out_dim * self.out_dim
		)

		self.deconvs = nn.ModuleList()

		for i in range(self.num_layers - 1):
			self.deconvs.append(
				nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
				)
		self.deconvs.append(
			nn.ConvTranspose2d(num_filters, obs_shape[0], 3, stride=2, output_padding=1)
		)

		# self.ln = nn.LayerNorm(self.feature_dim)

		self.outputs = dict()

	def forward(self, h):
		h = torch.relu(self.fc(h))
		self.outputs['fc'] = h

		deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
		self.outputs['deconv1'] = deconv

		for i in range(0, self.num_layers - 1):
			deconv = torch.relu(self.deconvs[i](deconv))
			self.outputs['deconv%s' % (i + 1)] = deconv

		obs = self.deconvs[-1](deconv)
		self.outputs['obs'] = obs

		return obs

	def log(self, L, step, log_freq):
		if step % log_freq != 0:
			return

		for k, v in self.outputs.items():
			L.log_histogram('train_decoder/%s_hist' % k, v, step)
			if len(v.shape) > 2:
				L.log_image('train_decoder/%s_i' % k, v[0], step)

		for i in range(self.num_layers):
			L.log_param(
				'train_decoder/deconv%s' % (i + 1), self.deconvs[i], step
			)
		L.log_param('train_decoder/fc', self.fc, step)

_AVAILABLE_DECODERS = {'pixel': PixelDecoder}

def make_decoder(
	decoder_type, obs_shape, feature_dim, num_layers, num_filters, num_shared_layers
):
	assert decoder_type in _AVAILABLE_DECODERS
	assert num_layers in OUT_DIM.keys(), 'invalid number of layers'
	if num_shared_layers == -1 or num_shared_layers == None:
		num_shared_layers = num_layers
	assert num_shared_layers <= num_layers and num_shared_layers > 0, \
		f'invalid number of shared layers, received {num_shared_layers} layers'
	return _AVAILABLE_DECODERS[decoder_type](
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

decoder = make_decoder('pixel', obs_shape, decoder_feature_dim, num_layers, num_filters, num_shared_layers,)
