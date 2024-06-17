import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image
from typing import List, Dict, Any


class MVSA(Dataset):
	def __init__(
		self, device: torch.device, annotations_file: str, data_dir: str, config: Dict[str, Any],
		transform=None, tokenizer=None, target_transform=None
	):
		self.device = device
		self.labels = pd.read_csv(annotations_file, keep_default_na=False)
		self.image_dir = os.path.join(data_dir, 'data')
		self.detection_dir = os.path.join(data_dir, f"features_{config['model']['rnet']}")
		self.text_model = config['model']['tnet']
		
		self.max_len = config['model']['max_len']
		self.transform = transform
		self.tokenizer = tokenizer
		self.target_transform = target_transform


	def __len__(self):
		return len(self.labels)


	def __getitem__(self, idx: int):
		# image
		image_path = os.path.join(self.image_dir, f'{self.labels.iloc[idx, 0]}.jpg')
		image = read_image(image_path)
		if self.transform:
			image = self.transform(image)
		
		# detection
		detection_path = os.path.join(self.detection_dir, f'{self.labels.iloc[idx, 0]}.pth')
		detection = torch.load(detection_path, map_location=self.device)
		
		# text
		text = self.labels.iloc[idx, 1]
		input_ids = torch.empty((self.max_len))
		attention_mask = torch.empty((self.max_len))
		if self.tokenizer:
			if self.text_model in ['bert_embedding', 'bert']:
				# add [CLS] and [SEP]
				encoding = self.tokenizer.encode_plus(
					text=text, add_special_tokens=True, padding='max_length', truncation=True,
					max_length=self.max_len, return_tensors='pt', 
					return_token_type_ids=False, return_attention_mask=True
				)
				input_ids = encoding['input_ids'].squeeze(0)
				attention_mask = encoding['attention_mask'].squeeze(0)
		
		# label: 'positive': 0, 'neutral': 1, 'negative': 2
		label = self.labels.iloc[idx, 2]
		if self.target_transform:
			label = self.target_transform(label)
		
		return image, detection, input_ids, attention_mask, label


	def collate_fn(self, batch: List[Any]):
		# support variable-sized image
		image = [item[0] for item in batch]
		
		# detection dict
		detection = [item[1] for item in batch]
		
		# text
		input_ids = [item[2] for item in batch]
		input_ids = torch.stack(input_ids)
		attention_mask = [item[3] for item in batch]
		attention_mask = torch.stack(attention_mask)
		
		# label
		label = [item[4] for item in batch]
		label = torch.LongTensor(label)
		
		return [image, detection, input_ids, attention_mask, label]