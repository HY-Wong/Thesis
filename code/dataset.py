import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor


class MVSA(Dataset):
	def __init__(self, annotations_file: str, data_dir: str, text_model: str, max_len: int, \
				 transform=None, tokenizer=None, target_transform=None):
		self.labels = pd.read_csv(annotations_file)
		self.data_dir = data_dir
		self.text_model = text_model
		self.max_len = max_len
		self.transform = transform
		self.tokenizer = tokenizer
		self.target_transform = target_transform


	def __len__(self):
		return len(self.labels)


	def __getitem__(self, idx):
		# text
		text = self.labels.iloc[idx, 1]
		input_ids = torch.empty((self.max_len))
		attention_mask = torch.empty((self.max_len))
		if self.tokenizer:
			if self.text_model == 'bert':
				# add [CLS] and [SEP]
				encoding = self.tokenizer.encode_plus(
					text=text, add_special_tokens=True, padding='max_length', truncation=True,
					max_length=self.max_len, return_tensors='pt', 
					return_token_type_ids=False, return_attention_mask=True
				)
				input_ids = encoding['input_ids'].squeeze(0)
				attention_mask = encoding['attention_mask'].squeeze(0)
		# image
		image_path = os.path.join(self.data_dir, '{}.jpg'.format(self.labels.iloc[idx, 0]))
		image = read_image(image_path)
		if self.transform:
			image = self.transform(image)
		# label: 'positive': 0, 'neutral': 1, 'negative': 2
		label = self.labels.iloc[idx, 2]
		if self.target_transform:
			label = self.target_transform(label)
		return image, input_ids, attention_mask, label