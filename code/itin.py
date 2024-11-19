import argparse
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import open_clip
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Dict, Any
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import dataset
import model

from configs import config


data_dir = '../data'
image_dir = '../image'
model_dir = '../model'

plt.rcParams['font.family'] = 'serif'

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
	"""
	Parse command-line arguments for training the ITIN model for multimodal sentiment analysis.
	"""
	parser = argparse.ArgumentParser(description='Train ITIN for Multimodal Sentiment Analysis')

	parser.add_argument(
		'--cfg', dest='cfg_file', help='config file', default='configs/itin_base.yaml', type=str
	)
	parser.add_argument(
		'--dataset', dest='dataset', help='possible datasets: mvsa_single, mvsa_multiple, climate_tv',
		default='mvsa_single', type=str
	)
	parser.add_argument(
		'--region', dest='region', help='Path to the directory containing region feature files.',
		default=None, type=str,
	)
	parser.add_argument(
		'--load_model', dest='load_model', help='load from a model checkpoint', default=None, type=str
	)
	parser.add_argument(
		'--training', dest='training', help='training the model or not',
		default=True, type=bool, action=argparse.BooleanOptionalAction
	)
	parser.add_argument(
		'--visual_baseline', dest='v_base', help='only use the visual encoder', 
		default=None, type=bool, action=argparse.BooleanOptionalAction
	)
	parser.add_argument(
		'--text_baseline', dest='t_base', help='only use the text encoder',
		default=None, type=bool, action=argparse.BooleanOptionalAction
	)
	parser.add_argument(
		'--cross_modal_baseline', dest='cm_base', help='only use the cross-modal interaction module',
		default=None, type=bool, action=argparse.BooleanOptionalAction
	)
	parser.add_argument(
		'--context_baseline', dest='c_base', help='only use the visual and text encoders',
		default=None, type=bool, action=argparse.BooleanOptionalAction
	)
	parser.add_argument(
		'--without_alignment', dest='wo_alignment',  help='without using the cross-modal alignment module',
		default=None, type=bool, action=argparse.BooleanOptionalAction
	)
	parser.add_argument(
		'--without_gating', dest='wo_gating', help='without using the cross-modal gating module',
		default=None, type=bool, action=argparse.BooleanOptionalAction
	)
	parser.add_argument(
		'--random_seed', dest='random_seed', help='set the random seed',
		default=1, type=int
	)
	parser.add_argument(
		'--optimizer', dest='optimizer', help='optimizer: adam, adam_w', default=None, type=str
	)
	parser.add_argument('--lr', dest='lr', help='learning rate', default=None, type=float)
	parser.add_argument('--weight_decay', dest='weight_decay', help='weight decay', default=None, type=float)
	parser.add_argument('--lambda', dest='_lambda', help='hyperparameter lambda', default=None, type=float)
	parser.add_argument('--num_regions', dest='num_regions', help='top regions based on scores', default=None, type=int)

	args = parser.parse_args()
	return args


def max_encoded_len(dataset_name: str, annotations_file: str, tokenizer):
	"""
	Calculate the maximum encoded sequence length for a given dataset.
	"""
	df = pd.read_csv(annotations_file, keep_default_na=False)
	max_len = 0

	for i in range(len(df)):
		text = "" if pd.isna(df.iloc[i, 1]) else df.iloc[i, 1]
		text_ids = tokenizer.encode(text, add_special_tokens=True)
		max_len = max(max_len, len(text_ids))
	
	print(f'{dataset_name} Dataset -- Max length: {max_len}')
	return max_len


def split_dataset(
	dataset: Dataset, generator: torch.Generator, batch_size: int, 
	train_ratio: float = 0.8, val_ratio: float = 0.1
):
	"""
	Split the given dataset into training, validation, and test sets based on specified ratios.
	"""
	train_size = int(train_ratio * len(dataset))
	val_size = int(val_ratio * len(dataset))
	test_size = len(dataset) - train_size - val_size

	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator)
	
	train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

	return train_loader, val_loader, test_loader


def plot_detection(
	image_path: str, detection_path: str, saved_path: str, fasterrcnn_label_names: List[Any], fasterrcnn_attributes: List[Any], 
	score_threshold: float = 0.8, attr_threshold: float = 0.1, g: torch.Tensor = None, num_regions: int = None
):
	"""
    Visualize bounding boxes and detected labels on the image.
    """
	image = Image.open(image_path)
	detection = torch.load(detection_path)
	
	mask = detection['scores'] > score_threshold

	boxes = detection['boxes'][mask]
	if isinstance(detection['labels'], list):
		labels = [item for item, m in zip(detection['labels'], mask.tolist()) if m]
	else:
		labels = detection['labels'][mask]
	scores = detection['scores'][mask]
	attrs = detection['attrs'][mask] if 'attrs' in detection else None
	attr_scores = detection['attr_scores'][mask] if 'attr_scores' in detection else None

	# select the objects with the highest m confidence scores
	if num_regions != None:
		boxes = boxes[:num_regions]
		labels = labels[:num_regions]
		scores = scores[:num_regions]
		if attrs != None:
			attrs = attrs[:num_regions]
		if attr_scores != None:
			attr_scores = attr_scores[:num_regions]

	pred_labels = []
	for i in range(boxes.shape[0]):
		if isinstance(labels, list):
			s = f'{labels[i]}: {scores[i]:.2f}'
		else:
			s = f'{fasterrcnn_label_names[labels[i]]}: {scores[i]:.2f}'
		if attrs != None and attr_scores[i] > attr_threshold:
			s = f'{fasterrcnn_attributes[attrs[i]]}' + ' ' + s
		if g != None:
			s = s + '\n' + f'g value: {g[i]:.4f}'
		pred_labels.append(s)
	pred_boxes = boxes.long().cpu()

	plt.imshow(image)

	for pred_box, pred_label in zip(pred_boxes, pred_labels):
		if pred_box[0] == 0:
			pred_box[0] = 1
		if pred_box[1] == 0:
			pred_box[1] = 1
		# bounding box
		plt.gca().add_patch(
			plt.Rectangle(
				(pred_box[0], pred_box[1]), pred_box[2] - pred_box[0], pred_box[3] - pred_box[1], 
				fill=False, edgecolor='red', linewidth=2, alpha=0.5
			)
		)
		# predicted label
		plt.gca().text(
			pred_box[0], pred_box[1] - 2, pred_label, bbox=dict(facecolor='blue', alpha=0.5),
			fontsize=8, color='white'
		)

	plt.savefig(saved_path, dpi=480)
	plt.close()


def get_activation(activation: Dict[str, torch.Tensor], name: str):
	"""
	Capture the output of a specific layer in a model during a forward pass.
	"""
	# the hook signature
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook


class FocalLoss(nn.Module):
	"""
	Implement the Focal Loss for addressing class imbalance in classification tasks.
	"""
	def __init__(self, alpha: torch.Tensor, gamma: float = 2.):
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
	

	def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
		ce_loss = nn.functional.cross_entropy(inputs, labels, reduction='none')
		pt = torch.exp(-ce_loss)
		loss = (self.alpha[labels] * (1 - pt) ** self.gamma * ce_loss).mean()
		return loss


def train(
	model: nn.Module, train_loader, val_loader, criterion, optimizer, scheduler, 
	writer, cfg, activation: Dict[str, torch.Tensor]
):
	epochs = cfg.training.epochs
	best_accuracy = 0.0
	
	for epoch in range(1, epochs + 1):
		# training
		model.train()

		running_loss = 0.
		
		desc = f'Epoch{epoch}/{epochs}'
		for data_ids, images, region_features, texts, attention_masks, labels in tqdm.tqdm(train_loader, desc=desc, leave=True):
			images = images.to(device)
			region_features = region_features.to(device)
			texts = texts.to(device)
			attention_masks = attention_masks.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			outputs = model(images, texts, attention_masks, region_features)
			loss = criterion(outputs, labels)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
			optimizer.step()

			running_loss += loss.item() * len(images)
			
		scheduler.step()

		avg_loss = running_loss / len(train_loader.dataset)

		# validation
		# disabling dropout and using population statistics for batch normalization
		model.eval()

		running_vloss = 0.
		all_labels = []
		all_predictions = []

		with torch.no_grad():
			for data_ids, images, region_features, texts, attention_masks, labels in val_loader:
				images = images.to(device)
				region_features = region_features.to(device)
				texts = texts.to(device)
				attention_masks = attention_masks.to(device)
				labels = labels.to(device)

				outputs = model(images, texts, attention_masks, region_features)
				loss = criterion(outputs, labels)
				_, predictions = torch.max(outputs, 1)
				all_labels.extend(labels.cpu().numpy())
				all_predictions.extend(predictions.cpu().numpy())

				running_vloss += loss.item() * len(images)

		avg_vloss = running_vloss / len(val_loader.dataset)
		accuracy = accuracy_score(all_labels, all_predictions)
		f1 = f1_score(all_labels, all_predictions, average='weighted')
		
		print(
			f'Epoch [{epoch:02d}], Train loss: {avg_loss:.4f}, Valid loss: {avg_vloss:.4f}	'\
			f'Valid Accuracy: {accuracy:.4f}, Valid F1 Score: {f1:.4f}'
		)
		writer.add_scalars(
			'Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss}, epoch
		)
		writer.add_scalars(
			'Validation Accuracy & F1 Score', {'Accuracy': accuracy, 'F1 Score': f1}, epoch
		)
		writer.flush()
		
		# run the model on the two examples shown in Fig. 7 of the ITIN paper
		with torch.no_grad():
			for data_ids, images, region_features, texts, attention_masks, labels in example_loader:
				images = images.to(device)
				region_features = region_features.to(device)
				texts = texts.to(device)
				attention_masks = attention_masks.to(device)
				labels = labels.to(device)

				outputs = model(images, texts, attention_masks, region_features)
				_, predictions = torch.max(outputs, 1)

				# save for visualizing gating values and attention scores to analyze 
				# the behavior of the Cross-Modal Gating Module and the Cross-Modal Alignment Module
				if accuracy > best_accuracy:
					if use_alignmnet:
						attn_values = activation['attn'].cpu()
						saving_path = os.path.join(model_dir, timestamp, 'attn_values.pt')
						torch.save(attn_values, saving_path)
					if use_gating:
						g_values = activation['g'].cpu()
						saving_path = os.path.join(model_dir, timestamp, 'g_values.pt')
						torch.save(g_values, saving_path)

		# track and save the best model
		if accuracy > best_accuracy:
			best_accuracy = accuracy
			model_path = os.path.join(model_dir, timestamp, 'model.pt')
			torch.save(model.state_dict(), model_path)
		# model_path = os.path.join(model_dir, timestamp, f'model{epoch:02d}.pt')
		# torch.save(model.state_dict(), model_path)

	writer.close()
	

def test(model: nn.Module, test_loader):
	# disabling dropout and using population statistics for batch normalization.
	model.eval()
	
	all_labels = []
	all_predictions = []

	with torch.no_grad():
		for data_ids, images, region_features, texts, attention_masks, labels in test_loader:
			images = images.to(device)
			region_features = region_features.to(device)
			texts = texts.to(device)
			attention_masks = attention_masks.to(device)
			labels = labels.to(device)

			outputs = model(images, texts, attention_masks, region_features)
			_, predictions = torch.max(outputs, 1)
			all_labels.extend(labels.cpu().numpy())
			all_predictions.extend(predictions.cpu().numpy())

	accuracy = accuracy_score(all_labels, all_predictions)
	f1 = f1_score(all_labels, all_predictions, average='weighted')

	print(f'Test Accuracy: {accuracy:.4f}')
	print(f'Test F1 Score: {f1:.4f}')
	print('Confusion Matrix')
	print(confusion_matrix(all_labels, all_predictions))


if __name__ == '__main__':
	args = parse_args()
	cfg = config.load_config(args.cfg_file)
	# override configuration with command line arguments if provided
	if args.region:
		cfg.model.region.model = args.region
	if args.v_base:
		cfg.model.visual.train_baseline = args.v_base
	if args.t_base:
		cfg.model.text.train_baseline = args.t_base
	if args.cm_base:
		cfg.model.cross_modal_baseline = args.cm_base
	if args.c_base:
		cfg.model.context_baseline = args.c_base
	if args.wo_alignment:
		cfg.model.without_alignment = args.wo_alignment
	if args.wo_gating:
		cfg.model.without_gating = args.wo_gating
	if args.optimizer:
		cfg.training.optimizer = args.optimizer
	if args.lr:
		cfg.training.lr = args.lr
	if args.weight_decay:
		cfg.training.weight_decay = args.weight_decay
	if args._lambda:
		cfg.model._lambda = args._lambda
	if args.num_regions:
		cfg.model.region.num_regions = args.num_regions

	# some ablation models do not compute attention scores or gating values
	use_alignmnet = True
	use_gating = True
	if cfg.model.visual.train_baseline or cfg.model.text.train_baseline or cfg.model.context_baseline:
		use_alignmnet = False
		use_gating = False
	if cfg.model.without_alignment:
		use_alignmnet = False
	if cfg.model.without_gating:
		use_gating = False

	# set the random seed for reproducibly splitting the dataset
	generator = torch.Generator().manual_seed(args.random_seed)

	# use the timestamp to store the trained model and analysis plots
	timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M') + f'_{args.dataset}'
	print(timestamp)
	if args.training:
		writer = SummaryWriter(log_dir=os.path.join(model_dir, timestamp, 'runs'))

	# image transformations
	if cfg.model.visual.model == 'resnet':
		transform = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	elif cfg.model.visual.model == 'clip':
		_, _, transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k') 

	# tokenizer
	if cfg.model.text.model == 'bert':
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
	elif cfg.model.text.model == 'clip':
		tokenizer = open_clip.get_tokenizer('ViT-B-32')

	# max_encoded_len('MVSA-Single', mvsa_single_annotations_file, tokenizer)
	# max_encoded_len('MVSA-Multiple', mvsa_multiple_annotations_file, tokenizer)

	# load datasets
	if args.dataset == 'mvsa_single':
		dataset_path = os.path.join(data_dir, 'MVSA_Single')
		batch_size = cfg.training.batch_size_mvsa_single
	elif args.dataset == 'mvsa_multiple':
		dataset_path = os.path.join(data_dir, 'MVSA_Multiple')
		batch_size = cfg.training.batch_size_mvsa_multiple
	elif args.dataset == 'climate_tv':
		dataset_path = os.path.join(data_dir, 'ClimateTV')
		batch_size = cfg.training.batch_size_climate_tv
	
	train_dataset = dataset.Dataset(
		'train.csv', dataset_path, cfg, 
		transform=transform, tokenizer=tokenizer
	)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
	val_dataset = dataset.Dataset(
		'val.csv', dataset_path, cfg, 
		transform=transform, tokenizer=tokenizer
	)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
	test_dataset = dataset.Dataset(
		'test.csv', dataset_path, cfg, 
		transform=transform, tokenizer=tokenizer
	)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
	# two examples shown in Fig. 7 of the ITIN paper
	example = dataset.Dataset(
		'example.csv', os.path.join(data_dir, 'Example'), cfg, 
		transform=transform, tokenizer=tokenizer
	)
	example_loader = DataLoader(example, batch_size=2, shuffle=False)

	# examine the first batch
	data_ids, images, region_features, texts, attention_masks, labels = next(iter(val_loader))
	print(f'Batch size: {len(data_ids)}')
	print(f'Image shape: {images.shape}')
	print(f'Region Features shape: {region_features.shape}')
	print(f'Text shape: {texts.shape}')
	print(f'Attention Mask shape: {attention_masks.shape}')
	print(f'Label shape: {labels.shape}')

	# plot one example
	# image_path = os.path.join(dataset_path, 'data', f'{data_ids[0]}.jpg')
	# image = Image.open(image_path) # load the image of original size
	# plt.imshow(image)
	# plt.show()

	# pre-trained Faster R-CNN
	# bua: https://github.com/MILVLG/bottom-up-attention.pytorch
	#	{boxes, region_features, labels, scores, attrs, attr_scores}
	# plot the detected objects of two examples shown in Fig. 7 of the ITIN paper
	fasterrcnn_label_names = []
	if cfg.model.region.model == 'fasterrcnn_bua':
		with open(os.path.join(data_dir, 'objects_vocab.txt')) as f:
			for label_name in f.readlines():
				fasterrcnn_label_names.append(label_name.split(',')[0].lower().strip())

	fasterrcnn_attributes = []
	if cfg.model.region.model == 'fasterrcnn_bua':
		with open(os.path.join(data_dir, 'attributes_vocab.txt')) as f:
			for attr in f.readlines():
				fasterrcnn_attributes.append(attr.split(',')[0].lower().strip())

	# data_ids, _, _, _, _, _ = next(iter(example_loader))
	# for i in range(len(data_ids)):
	# 	image_path = os.path.join(data_dir, 'Example', 'data', f'{data_ids[i]}.jpg')
	# 	detection_path = os.path.join(data_dir, 'Example', f"features_{cfg.model.region.model}", f'{data_ids[i]}.pth')
	# 	saved_path = os.path.join(image_dir, f"example{i:02d}_{cfg.model.region.model}.png")
	# 	plot_detection(
	# 		image_path, detection_path, saved_path, fasterrcnn_label_names, fasterrcnn_attributes, 
	# 		score_threshold=0.1, num_regions=6
	# 	)
	
	# train the ITIN model
	itin = model.ITIN(cfg, generator).to(device)

	if args.training:
		# register forward hooks on the layers of choice
		activation = {}
		if use_alignmnet:
			h1 = itin.cross_attn.softmax.register_forward_hook(get_activation(activation, 'attn'))
		if use_gating:
			h2 = itin.sigmoid.register_forward_hook(get_activation(activation, 'g'))
		
		# loss function
		criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
		# class_weights = torch.ones(cfg.model.num_classes)
		# criterion = FocalLoss(alpha=class_weights.to(device), gamma=cfg.training.gamma).to(device)
		
		# optimizer and scheduler
		# for name, param in itin.named_parameters():
		# 	if param.requires_grad:
		# 		print(name)
		params = [p for p in itin.parameters() if p.requires_grad]
		if cfg.training.optimizer == 'adam':
			optimizer = optim.Adam(params=params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
		elif cfg.training.optimizer == 'adam_w':
			optimizer = optim.AdamW(params=params, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
		scheduler = optim.lr_scheduler.StepLR(
			optimizer, step_size=cfg.training.lr_decay_step_size, gamma=cfg.training.lr_decay_factor
		)

		train(
			itin, train_loader, val_loader, criterion, optimizer, scheduler, writer, cfg, activation 
		)

		# detach the hooks
		if use_alignmnet:
			h1.remove()
		if use_gating:
			h2.remove()
	
	# test the ITIN model
	if args.training:
		model_path = os.path.join(model_dir, timestamp, 'model.pt')
		itin.load_state_dict(torch.load(model_path))
	if args.load_model:
		model_path = os.path.join(model_dir, args.load_model)
		itin.load_state_dict(torch.load(model_path))

	test(itin, test_loader)