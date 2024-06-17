import argparse
import datetime
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm
import yaml


from matplotlib.colors import Normalize
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image
from torchvision.transforms import v2 as T
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from typing import List, Dict, Union, Any

import dataset
import model


data_dir = '../data'
image_dir = '../image'
model_dir = '../model'

plt.rcParams['font.family'] = 'serif'


def parse_args():
	parser = argparse.ArgumentParser(description='Train ITIN for Multimodal Sentiment Analysis')

	parser.add_argument(
		'--cfg', dest='cfg_file', help='config file',
		default='configs/itin_base.yaml', type=str
	)
	parser.add_argument(
		'--dataset', dest='dataset', help='mvsa_single, mvsa_multiple',
		default='mvsa_single', type=str
	)
	parser.add_argument(
		'--vnet', dest='vnet', help='resnet, clip, vit, convnext',
		default=None, type=str
	)
	parser.add_argument(
		'--tnet', dest='tnet', help='bert_embedding, bert, clip',
		default=None, type=str
	)
	parser.add_argument(
		'--rnet', dest='rnet', help='fasterrcnn_pt, fasterrcnn_vg, fasterrcnn_bua',
		default=None, type=str
	)
	parser.add_argument(
		'--load_model', dest='load_model', help='load from a model checkpoint',
		default=None, type=str
	)
	parser.add_argument(
		'--training', dest='training', help='training the model or not',
		default=True, type=bool, action=argparse.BooleanOptionalAction
	)
	parser.add_argument(
		'--visual_baseline', dest='vbase', help='train the visual baseline model',
		default=None, type=bool
	)
	parser.add_argument(
		'--text_baseline', dest='tbase', help='train the textual baseline model',
		default=None, type=bool
	)
	parser.add_argument(
		'--cross_modal_only', dest='cross_modal_only', 
		help='train only with the cross-modal interaction module, excluding the visual and textual baselines',
		default=None, type=bool
	)
	parser.add_argument(
		'--linear_combination', dest='linear_combination', help='fuse the features using a linear combination',
		default=None, type=bool
	)
	parser.add_argument(
		'--temperature_scheduler', dest='schedule_t', help='linearly scheduling the softmax temperature',
		default=None, type=bool
	)
	parser.add_argument(
		'--temperature', dest='t', help='softmax temperature',
		default=None, type=float
	)
	parser.add_argument(
		'--starting_temperature', dest='start_t', help='starting softmax temperature',
		default=None, type=float
	)
	parser.add_argument(
		'--ending_temperature', dest='end_t', help='ending softmax temperature',
		default=None, type=float
	)
	parser.add_argument(
		'--scale_exponent', dest='scale_exponent', 
		help='exponent of the normalized scale for the sigmoid function',
		default=None, type=float
	)
	parser.add_argument(
		'--random_seed', dest='random_seed', help='set the random seed',
		default=1, type=int
	)

	args = parser.parse_args()
	return args


def get_transform():
	transforms = []
	transforms.append(T.ToDtype(torch.float, scale=True))
	return T.Compose(transforms)


def max_encoded_len(dataset_name: str, annotations_file: str, tokenizer):
	df = pd.read_csv(annotations_file, keep_default_na=False)
	max_len = 0

	for i in range(len(df)):
		text = "" if pd.isna(df.iloc[i, 1]) else df.iloc[i, 1]
		text_ids = tokenizer.encode(text, add_special_tokens=True)
		max_len = max(max_len, len(text_ids))
	
	print(f'{dataset_name} Dataset -- Max length: {max_len}')
	return max_len


def split_dataset(
	dataset: Dataset,  generator: torch.Generator, batch_size: int, 
	train_ratio: float = 0.8, val_ratio: float = 0.1
):
	# training set, validation set and test set is split by the ratio of 8:1:1
	train_size = int(train_ratio * len(dataset))
	val_size = int(val_ratio * len(dataset))
	test_size = len(dataset) - train_size - val_size

	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

	return train_loader, val_loader, test_loader


def plot_detection(
	image, detection, file_path: str, file_name: str,
	fasterrcnn_label_names: List[str], fasterrcnn_attributes: Union[List[str], None] = None, 
	score_threshold: float = 0.8, attr_threshold: float = 0.1, 
	g: torch.Tensor = None, a: torch.Tensor = None, m: int = None
):
	image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)

	mask = detection['scores'] > score_threshold

	boxes = detection['boxes'][mask]
	labels = detection['labels'][mask]
	scores = detection['scores'][mask]
	attrs = detection['attrs'][mask] if 'attrs' in detection else None
	attr_scores = detection['attr_scores'][mask] if 'attr_scores' in detection else None

	# select the top m region proposals
	if m != None:
		boxes = boxes[:m]
		labels = labels[:m]
		scores = scores[:m]
		if attrs != None:
			attrs = attrs[:m]
		if attr_scores != None:
			attr_scores = attr_scores[:m]

	pred_labels = []
	for i in range(boxes.shape[0]):
		s = f'{fasterrcnn_label_names[labels[i]]}: {scores[i]:.2f}'
		if attrs != None and attr_scores[i] > attr_threshold:
			s = f'{fasterrcnn_attributes[attrs[i]]}' + ' ' + s
		if g != None:
			s = s + '\n' + f'g value: {g[i]:.4f}'
		if a != None:
			s = s + '\n' + f'a value: {a[i]:.4f}'
		pred_labels.append(s)
	pred_boxes = boxes.long().cpu()

	plt.imshow(image.permute(1, 2, 0).cpu())

	for pred_box, pred_label in zip(pred_boxes, pred_labels):
		if pred_box[0] == 0:
			pred_box[0] = 1
		if pred_box[1] == 0:
			pred_box[1] = 1
		plt.gca().add_patch(
			plt.Rectangle(
				(pred_box[0], pred_box[1]), pred_box[2] - pred_box[0], pred_box[3] - pred_box[1], 
				fill=False, edgecolor='red', linewidth=2, alpha=0.5
			)
		)
		plt.gca().text(
			pred_box[0], pred_box[1] - 2, pred_label, bbox=dict(facecolor='blue', alpha=0.5),
			fontsize=8, color='white'
		)

	plt.savefig(os.path.join(file_path, f'{file_name}.png'), dpi=480)
	plt.close()


def plot_text(
	input_ids: torch.Tensor, attn:torch.Tensor, tokenizer, file_path: str, file_name: str
):
	# remove [CLS], [SEP], [PAD]
	mask = (input_ids != 0) & (input_ids != 101) & (input_ids != 102)
	tokens = tokenizer.convert_ids_to_tokens(input_ids[mask])
	text = tokenizer.decode(input_ids, skip_special_tokens=True)

	fig, axes = plt.subplots(attn.shape[0], 1, figsize=(10, 8))

	for i in range(attn.shape[0]):
		ax = axes[i]

		if torch.any(mask):
			# normalize the weights to the range [0, 1] and map weights to colors
			weights = attn[i][mask].cpu().numpy()
			norm = Normalize(vmin=weights.min(), vmax=weights.max())
			cmap = cm.Blues
			colors = cmap(norm(weights))

			# plot each word with its corresponding color
			for k, (token, color) in enumerate(zip(tokens, colors)):
				x = k % 5
				y = 20 - k // 5
				ax.text(x * 0.2 + 0.05, y * 0.05 - 0.05, token, color=color, fontsize=12, ha='left', va='center')

		ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
		ax.tick_params(axis='y', which='both', left=False, labelleft=False)
		ax.set_title(f'attn{i}', fontsize=12)
	
	fig.text(0.5, 0.05, text, ha='center', fontsize=10)
	plt.savefig(os.path.join(file_path, f'{file_name}.png'), dpi=480)
	plt.close()


def plot_values(predictions: torch.Tensor, labels: torch.Tensor, g: torch.Tensor, a: torch.Tensor, file_path: str):
		fig, ax1 = plt.subplots(figsize=(10, 6))

		x_values = range(1, predictions.shape[0] + 1)

		# plot the weights of the cross-modal components for each epoch
		ax1.plot(
			x_values, g[0].cpu().numpy(), label='g0', marker='o', color='tab:green', alpha=0.7
		)
		ax1.plot(
			x_values, g[1].cpu().numpy(), label='g1', marker='o', color='tab:blue', alpha=0.7
		)
		ax1.plot(
			x_values, a[0].cpu().numpy(), label='a0', marker='s', color='tab:green', alpha=0.7
		)
		ax1.plot(
			x_values, a[1].cpu().numpy(), label='a1', marker='s', color='tab:blue', alpha=0.7
		)

		ax1.set_xticks(x_values)
		ax1.set_xlabel('Epoch')
		ax1.set_ylabel('Value')

		# plot the predictions for each epoch
		ax2 = ax1.twinx()
		
		ax2.plot(
			x_values, predictions.cpu().numpy(),  
			label='prediction', marker='x', color='tab:orange', alpha=0.7
		)

		ax2.set_yticks([0, 1, 2])
		ax2.set_ylabel('Prediction')
		ax2.set_title(f'True label: {labels[0]}', fontsize=12)

		# add legends
		lines_1, labels_1 = ax1.get_legend_handles_labels()
		lines_2, labels_2 = ax2.get_legend_handles_labels()
		ax1.legend(lines_1 + lines_2, labels_1 + labels_2)

		plt.savefig(os.path.join(file_path, 'analysis.png'), dpi=480)
		plt.close()


def get_example(idx: int, config: Dict[str, Any], transform=None, tokenizer=None):
	example_dir = os.path.join(data_dir, 'Example')

	# image
	image_path = os.path.join(example_dir, f'example{idx:02d}.jpg')
	image = read_image(image_path)
	if transform:
		image = transform(image)
	
	# detection
	detection_path = os.path.join(example_dir, f'example{idx:02d}.pth')
	detection = torch.load(detection_path, map_location=device)
	
	# text
	text_path = os.path.join(example_dir, f'example{idx:02d}.txt')
	with open(text_path, 'r') as file:
		text = file.readline().strip()
	input_ids = torch.empty((config['model']['max_len']))
	attention_mask = torch.empty((config['model']['max_len']))
	if tokenizer:
		if config['model']['tnet'] in ['bert_embedding', 'bert']:
			# add [CLS] and [SEP]
			encoding = tokenizer.encode_plus(
				text=text, add_special_tokens=True, padding='max_length', truncation=True,
				max_length=config['model']['max_len'], return_tensors='pt', 
				return_token_type_ids=False, return_attention_mask=True
			)
			input_ids = encoding['input_ids'].squeeze(0)
			attention_mask = encoding['attention_mask'].squeeze(0)
	
	return image, detection, input_ids, attention_mask


def get_activation(activation: Dict[str, torch.Tensor], name: str):
	# the hook signature
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook


def train(
	model: nn.Module, train_loader, val_loader, criterion, optimizer, scheduler, writer, 
	config: Dict[str, Any], activation: Dict[str, torch.Tensor]
):
	best_vloss = 1e6
	m = config['model']['num_regions']
	epochs = config['training']['epochs']

	# store the registered values of the forward passes for each epoch for analysis
	first_batch_val_predictions = []
	first_batch_val_labels = []
	first_batch_val_g = []
	first_batch_val_a = []
	examples_predictions = []
	examples_labels = []
	examples_g = []
	examples_a = []
	
	for epoch in range(1, epochs + 1):
		# linear scheduler for the temperature scaler of the softmax function
		if config['model']['temperature']['schedule_t']:
			start_t = config['model']['temperature']['start_t']
			end_t = config['model']['temperature']['end_t']
			total_iters = config['model']['temperature']['total_iters']
			curr_t = start_t - (start_t - end_t) * (min(epoch, total_iters) - 1) / (total_iters - 1)
			model.t = curr_t
		
		# training
		model.train()

		running_loss = 0.
		first_batch = True
		
		desc = f'Epoch{epoch}/{epochs}'
		for images, detections, input_ids, attention_mask, labels in tqdm.tqdm(train_loader, desc=desc, leave=True):
			images = [image.to(device) for image in images]
			# select the top m region proposals
			region_features = torch.stack(
				[detection['region_features'][:m, :] for detection in detections]
			).to(device)
			input_ids = input_ids.to(device)
			attention_mask = attention_mask.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			outputs = model(images, input_ids, attention_mask, region_features)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * len(images)

			# visualize the gradient statistics of different feature components of the first batch of training data
			if first_batch:
				grad_W_Q = model.cross_attn.W_Q.weight.grad
				grad_W_K = model.cross_attn.W_K.weight.grad
				grad_fc_gating = model.fc_gating.weight.grad
				grad_fc_sentiment_v = torch.zeros((model.d_dim, model.d_dim))
				grad_fc_sentiment_t = torch.zeros((model.d_dim, model.d_dim))
				grad_fc_sentiment_c = torch.zeros((model.d_dim, model.d_dim))
				if not config['model']['cross_modal_only']:
					grad_first_fc_visual = model.mlp_sentiment_visual[0].weight.grad
					grad_first_fc_text = model.mlp_sentiment_text[0].weight.grad
					v_dim = model.visual_encoder.in_features
					t_dim = model.d_dim
					grad_fc_sentiment_v = grad_first_fc_visual[:, :v_dim]
					grad_fc_sentiment_t = grad_first_fc_text[:, :t_dim]
					grad_fc_sentiment_c = grad_first_fc_visual[:, v_dim:] + grad_first_fc_text[:, t_dim:]

				writer.add_scalars(
					'Gradient Mean', 
					{
						'W_Q': grad_W_Q.mean().item(), 
						'W_K': grad_W_K.mean().item(), 
						'fc_gating': grad_fc_gating.mean().item(),
						'fc_sentiment_v': grad_fc_sentiment_v.mean().item(),
						'fc_sentiment_t': grad_fc_sentiment_t.mean().item(),
						'fc_sentiment_c': grad_fc_sentiment_c.mean().item()
					}, 
					epoch
				)
				writer.add_scalars(
					'Gradient Std', 
					{
						'W_Q': grad_W_Q.std().item(), 
						'W_K': grad_W_K.std().item(), 
						'fc_gating': grad_fc_gating.std().item(),
						'fc_sentiment_v': grad_fc_sentiment_v.std().item(),
						'fc_sentiment_t': grad_fc_sentiment_t.std().item(),
						'fc_sentiment_c': grad_fc_sentiment_c.std().item()
					}, 
					epoch
				)
				writer.flush()

				first_batch = False
			
		scheduler.step()

		avg_loss = running_loss / len(train_loader.dataset)

		# validation
		# disabling dropout and using population statistics for batch normalization
		model.eval()

		running_vloss = 0.
		all_labels = []
		all_predictions = []
		first_batch = True

		with torch.no_grad():
			for images, detections, input_ids, attention_mask, labels in val_loader:
				images = [image.to(device) for image in images]
				# select the top m region proposals
				region_features = torch.stack(
					[detection['region_features'][:m, :] for detection in detections]
				).to(device)
				input_ids = input_ids.to(device)
				attention_mask = attention_mask.to(device)
				labels = labels.to(device)

				outputs = model(images, input_ids, attention_mask, region_features)
				loss = criterion(outputs, labels)
				_, predictions = torch.max(outputs, 1)
				all_labels.extend(labels.cpu().numpy())
				all_predictions.extend(predictions.cpu().numpy())

				running_vloss += loss.item() * len(images)

				# visualize the registered values of the first batch of validation data for each epoch
				if first_batch:
					g = activation['g']
					if config['model']['linear_combination']:
						a = model.linear_combine.detach().unsqueeze(0).repeat(len(images), 1)
					else:
						a = activation['a']

					first_batch_val_predictions.append(predictions)
					first_batch_val_labels.append(labels)
					first_batch_val_g.append(g)
					first_batch_val_a.append(a)
					
					for i in range(len(images)):
						plot_val_data_dir = os.path.join(model_dir, timestamp, f'data{i:02d}')
						os.makedirs(plot_val_data_dir, exist_ok=True)
						
						plot_detection(
							images[i], detections[i], plot_val_data_dir, f'epoch{epoch:02d}_detection',
							fasterrcnn_label_names, fasterrcnn_attributes, 
							score_threshold=0.1, g=g[i], a=a[i], m=m
						)
						plot_text(
							input_ids[i], activation['attn'][i], tokenizer,
							plot_val_data_dir, f'epoch{epoch:02d}_text'
						)
					
					first_batch = False

		avg_vloss = running_vloss / len(val_loader.dataset)
		accuracy = accuracy_score(all_labels, all_predictions)
		f1 = f1_score(all_labels, all_predictions, average='weighted')
		
		print(
			f'Epoch [{epoch:02d}], Train loss: {avg_loss:.4f}, Valid loss: {avg_vloss:.4f}	'\
			f'Valid Accuracy: {accuracy:.4f}, Test F1 Score: {f1:.4f}'
		)
		writer.add_scalars(
			'Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss}, epoch
		)
		writer.add_scalars(
			'Validation Accuracy & F1 Score', {'Accuracy': accuracy, 'F1 Score': f1}, epoch
		)
		writer.flush()

		# track and save the best model
		if avg_vloss < best_vloss:
			best_vloss = avg_vloss
			model_path = os.path.join(model_dir, timestamp, 'model.pt')
			torch.save(model.state_dict(), model_path)
		
		# run the model on the two examples shown in Fig. 7 of the ITIN paper
		with torch.no_grad():
			outputs = model(images_ex, input_ids_ex, attention_mask_ex, region_features_ex)
			_, predictions = torch.max(outputs, 1)

		g = activation['g']
		if config['model']['linear_combination']:
			a = model.linear_combine.detach().unsqueeze(0).repeat(len(images), 1)
		else:
			a = activation['a']

		examples_predictions.append(predictions)
		examples_labels.append(torch.tensor([1, 0]))
		examples_g.append(g)
		examples_a.append(a)

		for i in range(len(images_ex)):
			plot_ex_data_dir = os.path.join(model_dir, timestamp, f'example{i}')
			os.makedirs(plot_ex_data_dir, exist_ok=True)
					
			plot_detection(
				images_ex[i], detections_ex[i], plot_ex_data_dir, f'epoch{epoch:02d}_detection',
				fasterrcnn_label_names, fasterrcnn_attributes, 
				score_threshold=0.1, g=g[i], a=a[i], m=m
			)
			plot_text(
				input_ids_ex[i], activation['attn'][i], tokenizer,
				plot_ex_data_dir, f'epoch{epoch:02d}_text'
			)

	writer.close()
	
	# visualize data per batch for analysis
	first_batch_val_predictions = torch.stack(first_batch_val_predictions, dim=-1)
	first_batch_val_labels = torch.stack(first_batch_val_labels, dim=-1)
	first_batch_val_g = torch.stack(first_batch_val_g, dim=-1)
	first_batch_val_a = torch.stack(first_batch_val_a, dim=-1)

	for i in range(first_batch_val_predictions.shape[0]):
		plot_val_data_dir = os.path.join(model_dir, timestamp, f'data{i:02d}')
		os.makedirs(plot_val_data_dir, exist_ok=True)
		plot_values(
			first_batch_val_predictions[i], first_batch_val_labels[i], 
			first_batch_val_g[i], first_batch_val_a[i], plot_val_data_dir
		)
	
	examples_predictions = torch.stack(examples_predictions, dim=-1)
	examples_labels = torch.stack(examples_labels, dim=-1)
	examples_g = torch.stack(examples_g, dim=-1)
	examples_a = torch.stack(examples_a, dim=-1)

	for i in range(examples_predictions.shape[0]):
		plot_examples_data_dir = os.path.join(model_dir, timestamp, f'example{i}')
		os.makedirs(plot_examples_data_dir, exist_ok=True)
		plot_values(
			examples_predictions[i], examples_labels[i], 
			examples_g[i], examples_a[i], plot_examples_data_dir
		)
	

def test(model: nn.Module, test_loader, config: Dict[str, Any]):
	# disabling dropout and using population statistics for batch normalization.
	model.eval()
	
	all_labels = []
	all_predictions = []

	with torch.no_grad():
		for images, detections, input_ids, attention_mask, labels in test_loader:
			images = [image.to(device) for image in images]
			# select the top m region proposals
			region_features = torch.stack(
				[detection['region_features'][:config['model']['num_regions'], :] for detection in detections]
			).to(device)
			input_ids = input_ids.to(device)
			attention_mask = attention_mask.to(device)
			labels = labels.to(device)

			outputs = model(images, input_ids, attention_mask, region_features)
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
	with open(args.cfg_file, 'r') as file:
		config = yaml.safe_load(file)
	# override configuration with command line arguments if provided
	if args.vnet:
		config['model']['vnet'] = args.vnet
	if args.tnet:
		config['model']['tnet'] = args.tnet
	if args.rnet:
		config['model']['rnet'] = args.rnet
	if args.vbase:
		config['model']['vbase'] = args.vbase
	if args.tbase:
		config['model']['tbase'] = args.tbase
	if args.cross_modal_only:
		config['model']['cross_modal_only'] = args.cross_modal_only
	if args.linear_combination:
		config['model']['linear_combination'] = args.linear_combination
	if args.schedule_t:
		config['model']['temperature']['schedule_t'] = args.schedule_t
	if args.t:
		config['model']['temperature']['t'] = args.t
	if args.start_t:
		config['model']['temperature']['start_t'] = args.start_t
	if args.end_t:
		config['model']['temperature']['end_t'] = args.end_t
	if args.scale_exponent:
		config['model']['scale_exponent'] = args.scale_exponent
	# print(config)
	
	device = torch.device('cpu')
	if torch.cuda.is_available():
		device = torch.device('cuda:0')
	# if torch.backends.mps.is_available():
	# 	device = torch.device('mps')
	# print(device)

	# set the random seed for reproducibly splitting the dataset
	generator = torch.Generator().manual_seed(args.random_seed)

	# use the timestamp to store the trained model and analysis plots
	timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
	print(timestamp)
	if args.training:
		writer = SummaryWriter(log_dir=os.path.join(model_dir, timestamp, 'runs'))

	# datasets
	# MVSA-Single
	mvsa_single_data_dir = os.path.join(data_dir, 'MVSA_Single')
	mvsa_single_annotations_file = os.path.join(mvsa_single_data_dir, 'MVSA_Single.csv')
	# MVSA-Multiple
	mvsa_multiple_data_dir = os.path.join(data_dir, 'MVSA_Multiple')
	mvsa_multiple_annotations_file = os.path.join(mvsa_multiple_data_dir, 'MVSA_Multiple.csv')

	# apply image transformations
	transform = get_transform()

	# BERT tokenizer
	if config['model']['tnet'] in ['bert_embedding', 'bert']:
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) # bert-base-multilingual-cased

	# max_encoded_len('MVSA-Single', mvsa_single_annotations_file, tokenizer)
	# max_encoded_len('MVSA-Multiple', mvsa_multiple_annotations_file, tokenizer)

	# pre-trained Faster R-CNN
	# pt: https://github.com/pytorch/vision/tree/main/torchvision/models/detection
	#	{boxes, scores, labels, region_features}
	# vg: https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome
	# 	{boxes, scores, labels, region_features}
	# bua: https://github.com/MILVLG/bottom-up-attention.pytorch
	#	{boxes, region_features, labels, scores, attrs, attr_scores}
	mvsa_single = dataset.MVSA(
		device, mvsa_single_annotations_file, mvsa_single_data_dir, config, 
		transform=transform, tokenizer=tokenizer
	)
	mvsa_multiple = dataset.MVSA(
		device, mvsa_multiple_annotations_file, mvsa_multiple_data_dir, config, 
		transform=transform, tokenizer=tokenizer
	)

	if args.dataset == 'mvsa_single':
		train_loader, val_loader, test_loader = split_dataset(
			mvsa_single, generator, batch_size=config['training']['batch_size_mvsa_single']
		)
	elif args.dataset == 'mvsa_multiple':
		train_loader, val_loader, test_loader = split_dataset(
			mvsa_multiple, generator, batch_size=config['training']['batch_size_mvsa_multiple']
		)

	# examine the first batch
	images, detections, input_ids, attention_mask, labels = next(iter(test_loader))
	print(f'Image batch size: {len(images)}')
	print(f'Image shape: {images[0].shape}')
	for key, value in detections[0].items():
		print(f"Detections['{key}'] shape: {value.shape}")
	print(f'Input IDs batch shape: {input_ids.shape}')
	print(f'Attention Mask batch shape: {attention_mask.shape}')
	print(f'Label batch shape: {labels.shape}')

	# plot one example
	# image = images[0]
	# plt.imshow(image.permute(1, 2, 0))
	# plt.show()
	
	# plot some detected objects
	# Faster R-CNN pre-trained on COCO
	# pt: torchvision.models.detection.fasterrcnn_resnet50_fpn
	# Faster R-CNN pre-trained on Visual Genome
	# vg: https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome
	# bua: https://github.com/MILVLG/bottom-up-attention.pytorch
	if config['model']['rnet'] == 'fasterrcnn_pt':
		fasterrcnn_label_names = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta['categories']
	elif config['model']['rnet'] in ['fasterrcnn_vg', 'fasterrcnn_bua']:
		fasterrcnn_label_names = []
		with open(os.path.join(data_dir, 'objects_vocab.txt')) as f:
			for label_name in f.readlines():
				fasterrcnn_label_names.append(label_name.split(',')[0].lower().strip())

	if config['model']['rnet'] == 'fasterrcnn_bua':
		fasterrcnn_attributes = []
		with open(os.path.join(data_dir, 'attributes_vocab.txt')) as f:
			for attr in f.readlines():
				fasterrcnn_attributes.append(attr.split(',')[0].lower().strip())

	# load the data for the two examples as shown in Fig. 7 of the ITIN paper 
	image01, detection01, input_ids01, attention_mask01 = get_example(
		1, config, transform=transform, tokenizer=tokenizer
	)
	image02, detection02, input_ids02, attention_mask02 = get_example(
		2, config, transform=transform, tokenizer=tokenizer
	)
	# plot_detection(
	# 	image01, detection01, image_dir, 'example01_fasterrcnn_bua', 
	# 	fasterrcnn_label_names, fasterrcnn_attributes, score_threshold=0.1, m=config['model']['num_regions']
	# )
	# plot_detection(
	# 	image02, detection02, image_dir, 'example02_fasterrcnn_bua', 
	# 	fasterrcnn_label_names, fasterrcnn_attributes, score_threshold=0.1, m=config['model']['num_regions']
	# )

	images_ex = [image01.to(device), image02.to(device)]
	detections_ex = [detection01, detection02]
	# select the top m region proposals
	region_features_ex = torch.stack(
		[
			detection01['region_features'][:config['model']['num_regions'], :], 
			detection02['region_features'][:config['model']['num_regions'], :]
		]
	).to(device)
	input_ids_ex = torch.stack([input_ids01, input_ids02]).to(device)
	attention_mask_ex = torch.stack([attention_mask01, attention_mask02]).to(device)

	# train the ITIN model
	itin = model.ITIN(config).to(device)

	if args.training:
		# register forward hooks on the layers of choice
		activation = {}
		h1 = itin.cross_attn.softmax.register_forward_hook(get_activation(activation, 'attn'))
		h2 = itin.sigmoid.register_forward_hook(get_activation(activation, 'g'))
		if not config['model']['linear_combination']:
			h3 = itin.softmax.register_forward_hook(get_activation(activation, 'a'))

		# optimizer and scheduler
		# for name, param in itin.named_parameters():
		# 	if param.requires_grad:
		# 		print(name)
		params = [p for p in itin.parameters() if p.requires_grad]
		criterion = nn.CrossEntropyLoss()
		optimizer = optim.Adam(
			params=params, lr=config['training']['lr'], weight_decay=config['training']['weight_decay']
		)
		scheduler = optim.lr_scheduler.StepLR(
			optimizer, step_size=config['training']['lr_decay_step_size'], gamma=config['training']['lr_decay_factor']
		)

		train(
			itin, train_loader, val_loader, criterion, optimizer, scheduler, writer, config, activation 
		)

		# detach the hooks
		h1.remove()
		h2.remove()
		if not config['model']['linear_combination']:
			h3.remove()
	
	# test the ITIN model
	if args.training:
		model_path = os.path.join(model_dir, timestamp, 'model.pt')
		itin.load_state_dict(torch.load(model_path))
	if args.load_model:
		model_path = os.path.join(model_dir, args.load_model)
		itin.load_state_dict(torch.load(model_path))

	test(itin, test_loader, config)