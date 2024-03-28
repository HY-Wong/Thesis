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
writer = SummaryWriter(log_dir='runs/exp1')

data_dir = './../data'

# resize to resize_size=[256] using interpolation=InterpolationMode.BILINEAR
# central crop of crop_size=[224]
# rescale the values to [0.0, 1.0] and then normalize them using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
transform = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()

# BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# MVSA-Single dataset
# mvsa_single_data_dir = os.path.join(data_dir, 'MVSA_Single', 'data')
# mvsa_single_annotations_file = os.path.join(data_dir, 'MVSA_Single', 'MVSA_Single.csv')
# mvsa_single = dataset.MVSA(mvsa_single_annotations_file, mvsa_single_data_dir, transform=transform)

# MVSA-Multiple dataset
mvsa_multiple_data_dir = os.path.join(data_dir, 'MVSA_Multiple', 'data')
mvsa_multiple_annotations_file = os.path.join(data_dir, 'MVSA_Multiple', 'MVSA_Multiple.csv')
mvsa_multiple = dataset.MVSA(mvsa_multiple_annotations_file, mvsa_multiple_data_dir, transform=transform)


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
images, text_ids, labels = next(iter(train_loader))
print(f"Image batch shape: {images.size()}")
print(text_ids[0])
print(f"Label batch shape: {labels.size()}")

# print one example
# image = images[0].squeeze()
# plt.imshow(image.permute(1, 2, 0))
# plt.show()

tokens = tokenizer.encode(text_ids[0])
print(tokens)
encoding = tokenizer.encode_plus(
	text=text_ids,  # Preprocess sentence
	add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
	max_length=128,                  # Max length to truncate/pad
	pad_to_max_length=True,         # Pad sentence to max length
	return_tensors='pt',           # Return PyTorch tensor
	return_attention_mask=True      # Return attention mask
)


os.path.join(data_dir, 'MVSA_Multiple', 'MVSA_Multiple.csv')


def max_encoded_len(df_path: str, data_dir: str):
	df = pd.read_csv(df_path)
	max_len = 0

	for i in range(len(df)):
		text_path = os.path.join(data_dir, '{}.txt'.format(df.iloc[i, 0]))
		with open(text_path, 'rb') as file:
			text = file.readline().strip()
			try:
				text = text.decode('utf-8')
			except UnicodeDecodeError:
				print('UnicodeDecodeError: {}.txt'.format(df.iloc[i, 0]))
				text = ""
			text_ids = tokenizer.encode(text, add_special_tokens=True)
			max_len = max(max_len, len(text_ids))
	
	print(f'Max length: {max_len}')
	return max_len


max_len = max_encoded_len(mvsa_multiple_annotations_file, mvsa_multiple_data_dir)
exit(0)


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs: int = 10):
	for epoch in range(num_epochs):
		# training
		model.train()
		running_loss = 0.
		
		for images, text_ids, labels in train_loader:
			images, text_ids, labels = images.to(device), text_ids, labels.to(device)
			# images, text_ids, labels = images.to(device), text_ids.to(device), labels.to(device)

			optimizer.zero_grad()
			outputs =   model(images)
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
			for images, text_ids, labels in val_loader:
				images, text_ids, labels = images.to(device), text_ids, labels.to(device)
				# images, text_ids, labels = images.to(device), text_ids.to(device), labels.to(device)
				outputs = model(images)
				loss = criterion(outputs, labels)

				running_vloss += loss.item() * images.size(0)

		avg_vloss = running_vloss / len(val_loader.dataset)

		print(f'Epoch [{epoch+1}], Train loss: {avg_loss:.4f}, Valid loss: {avg_vloss:.4f}')
		# writer.add_scalars('Training vs. Validation Loss', { 'Training': avg_loss, 'Validation': avg_vloss }, epoch+1)
		# writer.flush()


def test(model, test_loader):
	model.eval()
	all_labels = []
	all_predictions = []

	with torch.no_grad():
		for images, text_ids, labels in test_loader:
			images, text_ids, labels = images.to(device), text_ids, labels.to(device)
			# images, text_ids, labels = images.to(device), text_ids.to(device), labels.to(device)

			outputs = model(images)
			_, predicted = torch.max(outputs, 1)
			all_labels.extend(labels.cpu().numpy())
			all_predictions.extend(predicted.cpu().numpy())

	accuracy = accuracy_score(all_labels, all_predictions)
	f1 = f1_score(all_labels, all_predictions, average='weighted')

	print(f"Test Accuracy: {accuracy:.4f}")
	print(f"Test F1 Score: {f1:.4f}")


visual_encoder = model.VisualEncoder(model='resnet18').to(device)

criterion = nn.CrossEntropyLoss()
params = [p for p in visual_encoder.parameters() if p.requires_grad]
optimizer = optim.Adam(params=params, lr=1e-3, weight_decay=1e-5)

train(visual_encoder, train_loader, val_loader, criterion, optimizer, 20)
test(visual_encoder, test_loader)





