import os
import torch
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image


class Dataset(Dataset):
	"""
	A custom dataset class for loading and processing image-text data with region-based features.
	"""
	def __init__(
		self, annotations_file: str, data_dir: str, cfg, transform=None, tokenizer=None
	):
		self.labels = pd.read_csv(os.path.join(data_dir, annotations_file), keep_default_na=False)
		self.image_dir = os.path.join(data_dir, 'data')
		self.detection_dir = os.path.join(data_dir, f'features_{cfg.model.region.model}')
		self.text_model = cfg.model.text.model
		
		self.num_regions = cfg.model.region.num_regions
		self.max_len = cfg.model.text.max_len
		self.transform = transform
		self.tokenizer = tokenizer


	def __len__(self):
		return len(self.labels)


	def __getitem__(self, idx: int):
		# data ID
		data_id = self.labels.iloc[idx, 0]

		# image
		image_path = os.path.join(self.image_dir, f'{self.labels.iloc[idx, 0]}.jpg')
		image = Image.open(image_path)
		if self.transform:
			image = self.transform(image)
		
		# detection
		detection_path = os.path.join(self.detection_dir, f'{self.labels.iloc[idx, 0]}.pth')
		detection = torch.load(detection_path)
		region_features = detection['region_features'][:self.num_regions, :]
		
		# text
		text = self.labels.iloc[idx, 1]
		if self.tokenizer:
			if self.text_model == 'bert':
				# add [CLS] and [SEP]
				encoding = self.tokenizer.encode_plus(
					text=text, add_special_tokens=True, padding='max_length', truncation=True,
					max_length=self.max_len, return_tensors='pt', 
					return_token_type_ids=False, return_attention_mask=True
				)
				text = encoding['input_ids'].squeeze(0)
				attention_mask = encoding['attention_mask'].squeeze(0)
			elif self.text_model == 'clip':
				text = self.tokenizer(text).squeeze(0)
				attention_mask = (text != 0).int()
		
		# label: 'positive': 0, 'neutral': 1, 'negative': 2
		label = self.labels.iloc[idx, 2]
		
		return data_id, image, region_features, text, attention_mask, label