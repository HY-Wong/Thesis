import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import boxes as box_ops, roi_align
from transformers import BertModel
from typing import List, Tuple, Union


class ObjectDetector(nn.Module):
	def __init__(self, model: str, freeze: bool = True, detections_per_img: int = 2):
		super().__init__()
		self.model = model

		if self.model == 'fasterrcnn':
			# https://github.com/pytorch/vision/tree/main/torchvision/models/detection
			fasterrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
			# print(fasterrcnn)

			# freeze the parameters
			if freeze:
				for param in fasterrcnn.parameters():
					param.requires_grad = False

			self.transform = fasterrcnn.transform
			self.backbone = fasterrcnn.backbone
			self.rpn = fasterrcnn.rpn

			# modify the RoIHeads 
			self.box_roi_pool = fasterrcnn.roi_heads.box_roi_pool
			self.box_head = fasterrcnn.roi_heads.box_head
			self.box_predictor = fasterrcnn.roi_heads.box_predictor

			self.box_coder = fasterrcnn.roi_heads.box_coder
			self.score_thresh = fasterrcnn.roi_heads.score_thresh
			self.nms_thresh = fasterrcnn.roi_heads.nms_thresh
			self.detections_per_img = detections_per_img
			self.out_features = fasterrcnn.roi_heads.box_head.fc7.out_features

	
	def postprocess_detections(
		self, class_logits: torch.Tensor, box_regression: torch.Tensor,  proposals: List[torch.Tensor], 
		image_shapes: List[Tuple[int, int]], box_features: torch.Tensor
	) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
		device = class_logits.device
		num_classes = class_logits.shape[-1]

		boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
		pred_boxes = self.box_coder.decode(box_regression, proposals)

		pred_scores = F.softmax(class_logits, -1)

		# splits the tensor into chunks
		pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
		pred_scores_list = pred_scores.split(boxes_per_image, 0)
		box_features_list = box_features.split(boxes_per_image, 0)

		all_boxes = []
		all_scores = []
		all_labels = []
		all_box_features = []
		for boxes, scores, image_shape, box_features in \
			zip(pred_boxes_list, pred_scores_list, image_shapes, box_features_list):
			boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

			# create labels for each prediction
			labels = torch.arange(num_classes, device=device)
			labels = labels.view(1, -1).expand_as(scores)

			# (1000, out_features) -> (1000, num_classes, out_features)
			box_features = box_features.unsqueeze(dim=1)
			box_features = box_features.expand(-1, num_classes, -1)
			
			# remove predictions with the background label
			boxes = boxes[:, 1:]
			scores = scores[:, 1:]
			labels = labels[:, 1:]
			box_features = box_features[:, 1:]

			# batch everything, by making every class prediction be a separate instance
			boxes = boxes.reshape(-1, 4)
			scores = scores.reshape(-1)
			labels = labels.reshape(-1)
			box_features = box_features.reshape(-1, 1024)

			# remove low scoring boxes
			inds = torch.where(scores > self.score_thresh)[0]
			boxes, scores, labels, box_features = boxes[inds], scores[inds], labels[inds], box_features[inds]

			# remove empty boxes
			keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
			boxes, scores, labels, box_features = boxes[keep], scores[keep], labels[keep], box_features[keep]

			# non-maximum suppression, independently done per class
			keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
			# keep only topk scoring predictions
			keep = keep[: self.detections_per_img]
			boxes, scores, labels, box_features = boxes[keep], scores[keep], labels[keep], box_features[keep]

			# ensure that each object has at least detections_per_img number of detections by append 
			# zeros tensors (empty detections) if if necessary
			if boxes.shape[0] < self.detections_per_img:
				num_fill = self.detections_per_img - boxes.shape[0]
				boxes = torch.cat(
					(boxes, torch.zeros(num_fill, boxes.shape[1], dtype=torch.float32, device=device))
				)
				scores = torch.cat(
					(scores, torch.zeros(num_fill, dtype=torch.float32, device=device))
				)
				labels = torch.cat(
					(labels, torch.zeros(num_fill, dtype=torch.int64, device=device))
				)
				box_features = torch.cat(
					(box_features, torch.zeros(num_fill, box_features.shape[1], dtype=torch.float32, device=device))
				)

			all_boxes.append(boxes)
			all_scores.append(scores)
			all_labels.append(labels)
			all_box_features.append(box_features)

		return all_boxes, all_scores, all_labels, all_box_features


	def forward(self, x: List[torch.Tensor]):
		original_image_sizes: List[Tuple[int, int]] = []
		for image in x:
			original_image_sizes.append(tuple(image.shape[-2:]))
		
		# GeneralizedRCNNTransform -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
		images, _ = self.transform(x)

		# BackboneWithFPN -> Dict[str, Tensor]
		# return layers of the network
		# fasterrcnn_resnet50_fpn: out_channels = 256
		features = self.backbone(images.tensors)

		# RegionProposalNetwork -> Tuple[List[Tensor], Dict[str, Tensor]
		# (batch_size, 1000, 4)
		proposals, _ = self.rpn(images, features)

		# RoIHeads
		# (batch_size * 1000, 256, 7, 7)
		box_features = self.box_roi_pool(features, proposals, images.image_sizes)
		# (batch_size * 1000, out_features)
		box_features = self.box_head(box_features)
		# (batch_size * 1000, num_classes), (batch_size * 1000, num_classes * 4)
		# fasterrcnn_resnet50_fpn: num_classes = 91
		class_logits, box_regression = self.box_predictor(box_features)
		# return detected objects 
		detections: List[Dict[str, torch.Tensor]] = []
		boxes, scores, labels, box_features = self.postprocess_detections(
			class_logits, box_regression, proposals, images.image_sizes, box_features
		)
		for i in range(len(boxes)):
			detections.append({'boxes': boxes[i], 'labels': labels[i], 'scores': scores[i]})
		# resize the images to original sizes
		detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
		
		x = torch.stack(box_features)
		return detections, x


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