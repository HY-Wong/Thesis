import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from typing import Tuple, Union


class VisualEncoder(nn.Module):
	def __init__(self, model: str, freeze=True, num_classes=3):
		super().__init__()
		self.num_classes = num_classes

		if model == 'resnet18':
			resnet18 = torchvision.models.resnet18(weights='IMAGENET1K_V1')

			# freeze the parameters
			if freeze:
				for param in resnet18.parameters():
					param.requires_grad = False

			modules = list(resnet18.children())[:-1]  # remove the last layer
			self.layers = nn.Sequential(*modules)
			self.in_features = resnet18.fc.in_features

		self.fc = nn.Linear(self.in_features, self.num_classes)

	def forward(self, x: torch.Tensor):
		x = self.layers(x)
		x = x.squeeze()
		x = self.fc(x)
		return x


class TextEncoder(nn.Module):
	def __init__(self, model: str):
		super().__init__()
		self.x = x

	def forward(self, x: torch.Tensor):
		return x


class ObjectDetector(nn.Module):
	def __init__(self):
		super().__init__()
		self.x = x

	def forward(self, x: torch.Tensor):
		return x


class CrossModalAlignmentModule(nn.Module):
	def __init__(self):
		super().__init__()
		self.x = x

	def forward(self, x: torch.Tensor):
		return x


class CrossModalGatingModule(nn.Module):
	def __init__(self):
		super().__init__()
		self.x = x

	def forward(self, x: torch.Tensor):
		return x


class ITIN(nn.Module):
	def __init__(self):
		super().__init__()
		self.x = x

	def forward(self, x: torch.Tensor):
		return x