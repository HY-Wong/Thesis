import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import boxes as box_ops, roi_align
from transformers import BertModel
from typing import List, Tuple, Union


class VisualEncoder(nn.Module):
	def __init__(self, model: str, freeze: bool = True):
		super().__init__()
		self.model = model

		if self.model == 'resnet18':
			resnet18 = torchvision.models.resnet18(weights='IMAGENET1K_V1')
			# print(resnet18)

			# freeze the parameters
			if freeze:
				for param in resnet18.parameters():
					param.requires_grad = False

			# 1. resize to resize_size=[256] using interpolation=InterpolationMode.BILINEAR
			# 2. central crop of crop_size=[224]
			# 3. rescale the values to [0.0, 1.0] and then normalize them using 
			#    mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
			self.transform = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()

			modules = list(resnet18.children())[:-1]  # remove the last layer
			self.layers = nn.Sequential(*modules)
			self.in_features = resnet18.fc.in_features


	def forward(self, x: List[torch.Tensor]):
		x = [self.transform(image) for image in x]
		x = torch.stack(x)

		# (batch_size, in_features) 
		# resnet18: in_features = 512
		x = self.layers(x)
		x = x.squeeze(dim=(2, 3))
		return x


class TextEncoder(nn.Module):
	def __init__(self, model: str, d_dim: int, freeze: bool = True):
		super().__init__()
		self.model = model
		self.d_dim = d_dim

		if self.model == 'bert':
			# may need to change the save directory
			bert = BertModel.from_pretrained('bert-base-multilingual-cased')
			# print(bert)

			# freeze the parameters
			if freeze:
				for param in bert.parameters():
					param.requires_grad = False

			self.layers = bert
			self.in_features = bert.pooler.dense.in_features

		# bidirectional GRU layer
		self.gru = nn.GRU(self.in_features, self.d_dim, bidirectional=True, batch_first=True)


	def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
		if self.model == 'bert':
			# attention_mask: avoid performing attention on padding token indices
			outputs = self.layers(input_ids=x, attention_mask=attn_mask, output_attentions=False)
			# (batch_size, seq_length, in_features)
			# bert: in_features = 768
			h = outputs['last_hidden_state']
			# h = h * attn_mask.unsqueeze(dim=-1)
		
		# bidirectional GRU
		# (batch_size, seq_length, hidden_dim * 2), (2, batch_size, hidden_dim)
		output, h = self.gru(h)
		h1 = output[..., :self.d_dim]
		h2 = output[..., self.d_dim:]
		# assert(torch.equal(h1[:, -1, :], h[0]))
		# assert(torch.equal(h2[:, 0, :], h[1]))
		
		# (batch_size, seq_length, hidden_dim)
		w = (h1 + h2) / 2
		# w = w * attn_mask.unsqueeze(dim=-1) # masking the padding tokens
		
		# (batch_size, hidden_dim)
		s = torch.mean(w, dim=1)
		# s = torch.sum(w, dim=1) / torch.sum(attn_mask, dim=1, keepdim=True) # masking the padding tokens 
		return w, s


class Attention(nn.Module):
	def __init__(self, q_dim: int, k_dim: int, embed_dim: int):
		super().__init__()
		self.q_dim = q_dim
		self.k_dim = k_dim
		self.embed_dim = embed_dim

		# cross attention without value-projection matrix
		self.W_Q = nn.Linear(self.q_dim ,self.embed_dim, bias=False)
		self.W_K = nn.Linear(self.k_dim ,self.embed_dim, bias=False)
		# self.W_V = nn.Linear(self.k_dim ,self.embed_dim, bias=False)
		self.softmax = nn.Softmax(dim=-1)
		self.scale = self.embed_dim ** -0.5


	def forward(self, r: torch.Tensor, w: torch.Tensor):
		# (batch_size, m, embed_dim)
		Q = self.W_Q(r)
		# (batch_size, seq_length, embed_dim)
		K = self.W_K(w)
		# (batch_size, m, seq_length)
		QK = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
		attn = self.softmax(QK)
		# (batch_size, m, d_dim)
		x = torch.matmul(attn, w)
		return x


class ITIN(nn.Module):
	def __init__(
		self, visual_model: str, text_model: str, rf_dim: int, d_dim: int, k_dim: int, num_classes: int = 3, 
		_lambda: int = 0.2, visual_baseline: bool = False, text_baseline: bool = False
	):
		super().__init__()
		self.rf_dim = rf_dim
		self.d_dim = d_dim
		self.k_dim = k_dim
		self.num_classes = num_classes
		self._lambda = _lambda
		self.visual_baseline = visual_baseline
		self.text_baseline = text_baseline

		self.visual_encoder = VisualEncoder(model=visual_model)
		self.text_encoder = TextEncoder(model=text_model, d_dim=d_dim)
		# linear project to a d-dimentional regional feature
		self.fc_region = nn.Linear(self.rf_dim, self.d_dim)

		if self.visual_baseline:
			self.fc_sentiment_visual = nn.Linear(self.visual_encoder.in_features, self.num_classes)

		if self.text_baseline:
			self.fc_sentiment_text = nn.Linear(self.d_dim, self.num_classes)
		
		# Cross-Modal Alignment Module
		self.cross_attn = Attention(q_dim=self.d_dim, k_dim=self.d_dim, embed_dim=self.k_dim)
		
		# Cross-Modal Gating Module
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=-1)
		self.relu = nn.ReLU()
		self.fc_gating = nn.Linear(self.d_dim * 2, self.d_dim)
		self.mlp_gating = nn.Sequential(
			nn.Linear(self.d_dim, self.d_dim),
			nn.ReLU(),
			nn.Linear(self.d_dim, 1),
			nn.ReLU()
		)

		# Multimodal Sentiment Classification
		self.mlp_sentiment_visual = nn.Sequential(
			nn.Linear(self.visual_encoder.in_features + self.d_dim, self.d_dim),
			nn.ReLU(),
			nn.Linear(self.d_dim, d_dim),
			nn.ReLU()
		)
		self.mlp_sentiment_text = nn.Sequential(
			nn.Linear(self.d_dim * 2, self.d_dim),
			nn.ReLU(),
			nn.Linear(self.d_dim, self.d_dim),
			nn.ReLU()
		)
		self.fc_sentiment = nn.Linear(self.d_dim, self.num_classes)

	# def initialize_parameters(self):
	# 	# todo: suitable parameters initialization of each component
	# 	pass


	def forward(
		self, images: List[torch.Tensor], input_ids: torch.Tensor, attention_mask: torch.Tensor, 
		region_features: torch.Tensor
	):
		# (batch_size, in_features) 
		v = self.visual_encoder(images)
		if self.visual_baseline:
			# (batch_size, num_classes) 
			x = self.fc_sentiment_visual(v)
			return x
		
		# (batch_size, seq_length, d_dim), (batch_size, d_dim)
		w, s = self.text_encoder(input_ids, attention_mask)
		if self.text_baseline:
			# (batch_size, num_classes) 
			x = self.fc_sentiment_text(s)
			return x
		
		# (batch_size, m, d_dim)
		r = self.fc_region(region_features)
		
		# Cross-Modal Alignment Module
		# (batch_size, m, d_dim)
		u = self.cross_attn(r, w)
		
		# Cross-Modal Gating Module
		# (batch_size, m)
		g = self.sigmoid(torch.sum(r * u, dim=-1))
		# (batch_size, m, d_dim * 2)
		c = g.unsqueeze(dim=-1) * torch.cat((r, u), dim=-1)
		# (batch_size, m, d_dim)
		o = self.relu(self.fc_gating(c))
		z = o + r
		# (batch_size, m)
		a = self.relu(self.mlp_gating(z))
		a = a.squeeze(dim=-1)
		a = self.softmax(a)
		a = a.unsqueeze(dim=-1)
		# (batch_size, d_dim)
		c = torch.matmul(z.transpose(-2, -1), a).squeeze(dim=-1)

		# Multimodal Sentiment Classification
		# (batch_size, d_dim)
		f1 = self.mlp_sentiment_visual(torch.cat((v, c), dim=-1))
		# (batch_size, d_dim)
		f2 = self.mlp_sentiment_text(torch.cat((s, c), dim=-1))
		# (batch_size, d_dim)
		f = self._lambda * f1 + (1 - self._lambda) * f2
		# (batch_size, num_classes)
		x = self.fc_sentiment(f)
		# return gating value g for inspection
		return x, g