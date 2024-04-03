import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score

import dataset
import model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# using TensorBoard 
# writer = SummaryWriter(log_dir='runs/exp1')

data_dir = './../data'
# MVSA-Single
mvsa_single_data_dir = os.path.join(data_dir, 'MVSA_Single', 'data')
mvsa_single_annotations_file = os.path.join(data_dir, 'MVSA_Single', 'MVSA_Single.csv')
# MVSA-Multiple
mvsa_multiple_data_dir = os.path.join(data_dir, 'MVSA_Multiple', 'data')
mvsa_multiple_annotations_file = os.path.join(data_dir, 'MVSA_Multiple', 'MVSA_Multiple.csv')

# 1. resize to resize_size=[256] using interpolation=InterpolationMode.BILINEAR
# 2. central crop of crop_size=[224]
# 3. rescale the values to [0.0, 1.0] and then normalize them using 
#    mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
transform = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()

# BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def max_encoded_len(dataset_name: str, annotations_file: str):
	df = pd.read_csv(annotations_file)
	max_len = 0

	for i in range(len(df)):
		text = "" if pd.isna(df.iloc[i, 1]) else df.iloc[i, 1]
		text_ids = tokenizer.encode(text, add_special_tokens=True)
		max_len = max(max_len, len(text_ids))
	
	print(f'{dataset_name} Dataset -- Max length: {max_len}')
	return max_len


# max_encoded_len('MVSA-Single', mvsa_single_annotations_file)
# max_encoded_len('MVSA-Multiple', mvsa_multiple_annotations_file)

mvsa_single = dataset.MVSA(
	mvsa_single_annotations_file, mvsa_single_data_dir, text_model='bert', 
	max_len=128, transform=transform, tokenizer=tokenizer
)
mvsa_multiple = dataset.MVSA(
	mvsa_multiple_annotations_file, mvsa_multiple_data_dir, text_model='bert', 
	max_len=128, transform=transform, tokenizer=tokenizer
)


def split_dataset(dataset: Dataset,  batch_size: int, train_ratio: float = 0.8, val_ratio: float = 0.1):
	# training set, validation set and test set is split by the ratio of 8:1:1
	train_size = int(train_ratio * len(dataset))
	val_size = int(val_ratio * len(dataset))
	test_size = len(dataset) - train_size - val_size

	train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, val_loader, test_loader


# train_loader, val_loader, test_loader = split_dataset(mvsa_single, batch_size=64)
train_loader, val_loader, test_loader = split_dataset(mvsa_multiple, batch_size=128)

# examine the first batch
images, input_ids, attention_mask, labels = next(iter(train_loader))
print(f"Image batch shape: {images.shape}")
print(f"Input IDs batch shape: {input_ids.shape}")
print(f"Attention Mask batch shape: {attention_mask.shape}")
print(f"Label batch shape: {labels.shape}")

# print one example
# image = images[0].squeeze()
# plt.imshow(image.permute(1, 2, 0))
# plt.show()


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs: int = 10):
	for epoch in range(num_epochs):
		# training
		model.train()
		running_loss = 0.
		
		for images, input_ids, attention_mask, labels in train_loader:
			images = images.to(device)
			input_ids = input_ids.to(device)
			attention_mask = attention_mask.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			# outputs = model(images)
			outputs = model(input_ids, attention_mask)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item() * images.size(0)

		avg_loss = running_loss / len(train_loader.dataset)
		
		# validation
		# disabling dropout and using population statistics for batch normalization.
		model.eval()
		running_vloss = 0.

		with torch.no_grad():
			for images, input_ids, attention_mask, labels in val_loader:
				images = images.to(device)
				input_ids = input_ids.to(device)
				attention_mask = attention_mask.to(device)
				labels = labels.to(device)

				# outputs = model(images)
				outputs = model(input_ids, attention_mask)
				loss = criterion(outputs, labels)

				running_vloss += loss.item() * images.size(0)

		avg_vloss = running_vloss / len(val_loader.dataset)

		print(f'Epoch [{epoch+1}], Train loss: {avg_loss:.4f}, Valid loss: {avg_vloss:.4f}')
		# writer.add_scalars(
		# 	'Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss}, epoch+1
		# )
		# writer.flush()


def test(model, test_loader):
	model.eval()
	all_labels = []
	all_predictions = []

	with torch.no_grad():
		for images, input_ids, attention_mask, labels in test_loader:
			images = images.to(device)
			input_ids = input_ids.to(device)
			attention_mask = attention_mask.to(device)
			labels = labels.to(device)

			# outputs = model(images)
			outputs = model(input_ids, attention_mask)
			_, predicted = torch.max(outputs, 1)
			all_labels.extend(labels.cpu().numpy())
			all_predictions.extend(predicted.cpu().numpy())

	accuracy = accuracy_score(all_labels, all_predictions)
	f1 = f1_score(all_labels, all_predictions, average='weighted')

	print(f"Test Accuracy: {accuracy:.4f}")
	print(f"Test F1 Score: {f1:.4f}")


# visual_encoder = model.VisualEncoder(model='resnet18').to(device)

# criterion = nn.CrossEntropyLoss()
# params = [p for p in visual_encoder.parameters() if p.requires_grad]
# optimizer = optim.Adam(params=params, lr=1e-3, weight_decay=1e-5)

# train(visual_encoder, train_loader, val_loader, criterion, optimizer, 1)
# test(visual_encoder, test_loader)

text_encoder = model.TextEncoder(model='bert', hidden_dim=256).to(device)

criterion = nn.CrossEntropyLoss()
params = [p for p in text_encoder.parameters() if p.requires_grad]
optimizer = optim.Adam(params=params, lr=1e-3, weight_decay=1e-5)

train(text_encoder, train_loader, val_loader, criterion, optimizer, 1)
test(text_encoder, test_loader)