import torch
import torch.nn as nn
import torchvision

from transformers import BertModel
from typing import List, Dict, Tuple, Union, Any


class VisualEncoder(nn.Module):
	def __init__(self, model: str):
		super().__init__()
		self.model = model

		if self.model == 'resnet':
			resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
			# print(resnet)
			self.in_features = 512
			# 1. resize to resize_size=[256] using interpolation=InterpolationMode.BILINEAR
			# 2. central crop of crop_size=[224]
			# 3. rescale the values to [0.0, 1.0] and then normalize them using 
			#    mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
			self.transform = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
			modules = list(resnet.children())[:-1]  # remove the last layer
			self.layers = nn.Sequential(*modules)

		# freeze the parameters
		for param in self.layers.parameters():
			param.requires_grad = False


	def forward(self, x: List[torch.Tensor]):
		x = [self.transform(image) for image in x]
		x = torch.stack(x)

		# (batch_size, in_features) 
		# resnet: in_features = 512
		x = self.layers(x)
		x = x.squeeze(dim=(2, 3))
		return x


class TextEncoder(nn.Module):
	def __init__(self, model: str, d_dim: int):
		super().__init__()
		self.model = model
		self.d_dim = d_dim

		if self.model == 'bert':
			bert = BertModel.from_pretrained('bert-base-uncased')
			# print(bert)
			self.in_features = 768
			self.layers = bert
		elif self.model == 'bert_embedding':
			bert = BertModel.from_pretrained('bert-base-uncased')
			# print(bert)
			self.in_features = 768
			self.layers = bert.embeddings

		# freeze the parameters
		for param in self.layers.parameters():
			param.requires_grad = False
			
		self.gru = nn.GRU(self.in_features, self.d_dim, bidirectional=True, batch_first=True)


	def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
		if self.model == 'bert':
			# using the last hidden state of BERT as an embedding
			# (batch_size, seq_length, in_features)
			# bert: in_features = 768
			outputs = self.layers(input_ids=x, attention_mask=attn_mask, output_attentions=False)
			x = outputs['last_hidden_state']
		elif self.model == 'bert_embedding':
			# using the embedding layer of BERT as an embedding
			# (batch_size, seq_length, in_features)
			# bert: in_features = 768
			x = self.layers(input_ids=x)
		
		# bidirectional GRU
		# (batch_size, seq_length, d_dim * 2), (2, batch_size, d_dim)
		output, h = self.gru(x)
		h1 = output[..., :self.d_dim]
		h2 = output[..., self.d_dim:]
		# assert(torch.equal(h1[:, -1, :], h[0]))
		# assert(torch.equal(h2[:, 0, :], h[1]))
		
		# word-level context feature
		# (batch_size, seq_length, d_dim)
		w = (h1 + h2) / 2
		# sentence-level context feature
		# (batch_size, d_dim)
		s = torch.mean(w, dim=1)
		return w, s


class Attention(nn.Module):
	def __init__(self, q_dim: int, k_dim: int, embed_dim: int, t: int):
		super().__init__()
		self.q_dim = q_dim
		self.k_dim = k_dim
		self.embed_dim = embed_dim
		self.t = t

		# cross attention without value-projection matrix
		self.W_Q = nn.Linear(self.q_dim ,self.embed_dim, bias=False)
		self.W_K = nn.Linear(self.k_dim ,self.embed_dim, bias=False)
		self.softmax = nn.Softmax(dim=-1)
		self.scale = self.embed_dim ** -0.5


	def forward(self, r: torch.Tensor, w: torch.Tensor):
		# (batch_size, m, embed_dim)
		Q = self.W_Q(r)
		# (batch_size, seq_length, embed_dim)
		K = self.W_K(w)
		# (batch_size, m, seq_length)
		QK = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
		attn = self.softmax(QK / self.t)
		# (batch_size, m, d_dim)
		x = torch.matmul(attn, w)
		return x


class ITIN(nn.Module):
	def __init__(self, config: Dict[str, Any]):
		super().__init__()
		self.visual_baseline = config['model']['vbase']
		self.text_baseline = config['model']['tbase']
		self.cross_modal_only = config['model']['cross_modal_only']
		self.linear_combination = config['model']['linear_combination']
		self.rf_dim = config['model']['rf_dim']
		self.d_dim = config['model']['d_dim']
		self.k_dim = config['model']['k_dim']
		self._lambda = config['model']['lambda']
		self.t = config['model']['temperature']['t']
		self.num_classes = config['model']['num_classes']
		self.m = config['model']['num_regions']
		self.scale = self.d_dim ** config['model']['scale_exponent']
		
		# context information extraction
		self.visual_encoder = VisualEncoder(model=config['model']['vnet'])
		self.text_encoder = TextEncoder(model=config['model']['tnet'], d_dim=self.d_dim)
		# linear project to a d-dimentional regional feature
		self.fc_region = nn.Linear(self.rf_dim, self.d_dim)

		if self.visual_baseline:
			self.fc_sentiment_visual = nn.Linear(self.visual_encoder.in_features, self.num_classes)

		if self.text_baseline:
			self.fc_sentiment_text = nn.Linear(self.d_dim, self.num_classes)
		
		# cross-modal alignment module
		self.cross_attn = Attention(self.d_dim, self.d_dim, self.k_dim, self.t)
		
		# cross-modal gating module
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=-1)
		self.relu1 = nn.ReLU()
		self.fc_gating = nn.Linear(self.d_dim * 2, self.d_dim)
		

		if self.linear_combination:
			# use the same weight for each region to obtain a fused alignment feature for each data point
			self.linear_combine = nn.Parameter(torch.ones(self.m) / self.m)
			self.relu2 = nn.ReLU()
		else:
			# use region-feature dependent weights to obtain a fused alignment feature for each data point
			self.mlp_gating = nn.Sequential(
				nn.Linear(self.d_dim, self.d_dim),
				nn.ReLU(),
				nn.Linear(self.d_dim, 1),
				nn.ReLU()
			)

		if self.cross_modal_only:
			self.fc_sentiment_cross_modal = nn.Linear(self.d_dim, self.num_classes)

		# multimodal sentiment classification
		self.mlp_sentiment_visual = nn.Sequential(
			nn.Linear(self.visual_encoder.in_features + self.d_dim, self.d_dim),
			nn.ReLU(),
			nn.Linear(self.d_dim, self.d_dim),
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
		# context information extraction
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
		
		# cross-modal alignment module
		# (batch_size, m, d_dim)
		u = self.cross_attn(r, w)
		
		# cross-modal gating module
		# (batch_size, m)
		g = self.sigmoid(torch.sum(r * u, dim=-1) * self.scale)
		# (batch_size, m, d_dim * 2)
		c = g.unsqueeze(dim=-1) * torch.cat((r, u), dim=-1)
		# (batch_size, m, d_dim)
		o = self.relu1(self.fc_gating(c))
		z = o + r
		
		if self.linear_combination:
			# (batch_size, d_dim)
			c = self.relu2(torch.matmul(z.transpose(-2, -1), self.linear_combine).squeeze(dim=-1))
		else:
			# (batch_size, m)
			a = self.mlp_gating(z)
			a = a.squeeze(dim=-1)
			a = self.softmax(a)
			a = a.unsqueeze(dim=-1)
			# (batch_size, d_dim)
			c = torch.matmul(z.transpose(-2, -1), a).squeeze(dim=-1)

		if self.cross_modal_only:
			# (batch_size, num_classes) 
			x = self.fc_sentiment_cross_modal(c)
			return x
		
		# multimodal sentiment classification
		# (batch_size, d_dim)
		f1 = self.mlp_sentiment_visual(torch.cat((v, c), dim=-1))
		# (batch_size, d_dim)
		f2 = self.mlp_sentiment_text(torch.cat((s, c), dim=-1))
		# (batch_size, d_dim)
		f = self._lambda * f1 + (1 - self._lambda) * f2
		# (batch_size, num_classes)
		x = self.fc_sentiment(f)
		return x