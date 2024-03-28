import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor


class MVSA(Dataset):
	def __init__(self, annotations_file, data_dir, transform=None, tokenizer=None, target_transform=None):
		self.labels = pd.read_csv(annotations_file)
		self.data_dir = data_dir
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		# text
		text_path = os.path.join(self.data_dir, '{}.txt'.format(self.labels.iloc[idx, 0]))
		with open(text_path, 'rb') as file:
			text = file.readline().strip()
			try:
				text = text.decode('utf-8')
			except UnicodeDecodeError:
				print('UnicodeDecodeError: {}.txt'.format(self.labels.iloc[idx, 0]))
				text = ""
		# image
		image_path = os.path.join(self.data_dir, '{}.jpg'.format(self.labels.iloc[idx, 0]))
		image = read_image(image_path)
		if self.transform:
			image = self.transform(image)
		# label: 'positive': 0, 'neutral': 1, 'negative': 2
		label = self.labels.iloc[idx, 1]
		if self.target_transform:
			label = self.target_transform(label)
		return image, text, label




