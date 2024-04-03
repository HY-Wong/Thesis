import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from transformers import BertModel
from typing import Tuple, Union


class VisualEncoder(nn.Module):
	def __init__(self, model: str, freeze=True, num_classes=3):
		super().__init__()
		self.num_classes = num_classes
		self.model = model

		if self.model == 'resnet18':
			resnet18 = torchvision.models.resnet18(weights='IMAGENET1K_V1')
			# print(resnet18)

			# freeze the parameters
			if freeze:
				for param in resnet18.parameters():
					param.requires_grad = False

			modules = list(resnet18.children())[:-1]  # remove the last layer
			self.layers = nn.Sequential(*modules)
			self.in_features = resnet18.fc.in_features

		self.fc = nn.Linear(self.in_features, self.num_classes)


	def forward(self, x: torch.Tensor):
		# (batch_size, in_features) resnet18: in_features=512
		x = self.layers(x)
		x = x.squeeze()

		# (batch_size, num_classes)
		x = self.fc(x)
		return x


class TextEncoder(nn.Module):
	def __init__(self, model: str, hidden_dim: int, freeze=True, num_classes=3):
		super().__init__()
		self.num_classes = num_classes
		self.model = model
		self.hidden_dim = hidden_dim

		if self.model == 'bert':
			# may need to change the save directory
			bert = BertModel.from_pretrained('bert-base-uncased')
			# print(bert)

			# freeze the parameters
			if freeze:
				for param in bert.parameters():
					param.requires_grad = False

			self.layers = bert
			self.in_features = bert.pooler.dense.in_features

		# bidirectional GRU layer
		self.gru = nn.GRU(self.in_features, self.hidden_dim, bidirectional=True, batch_first=True)
		
		self.fc = nn.Linear(self.hidden_dim, self.num_classes)


	def forward(self, x: torch.Tensor, attn_mask: Union[torch.Tensor, None] = None):
		if self.model == 'bert':
			# attention_mask: avoid performing attention on padding token indices
			outputs = self.layers(input_ids=x, attention_mask=attn_mask, output_attentions=False)
			# (batch_size, seq_length, in_features)
			# bert: in_features = 768
			h = outputs['last_hidden_state']
			# (batch_size, in_features)
			o = outputs['pooler_output']
		
		# bidirectional GRU
		# (2, batch_size, hidden_dim)
		_, h = self.gru(h)
		h = (h[0] + h[1]) / 2

		# (batch_size, num_classes)
		x = self.fc(h)
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


	def initialize_parameters(self):
		# todo: suitable parameters initialization of each component
		pass


	def forward(self, x: torch.Tensor):
		return x