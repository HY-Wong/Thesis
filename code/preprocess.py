import os
import re
import json
import numpy as np
import pandas as pd

from collections import defaultdict
from typing import Union
from torchvision.io import read_image


def combine_text_image_label(text_label: Union[str, None], image_label: Union[str, None]):
	"""
	Generate a single sentiment label for each image-text pair.
	"""
	# invalid labels
	if text_label == None or image_label == None:
		return 'invalid'
	# consistent labels
	if text_label == image_label:
		return text_label
	# positive / negative & neutral -> positive / negative
	if text_label == 'neutral':
		return image_label
	if image_label == 'neutral':
		return text_label
	# inconsistent labels
	return 'inconsistent'


def remove_url(text: str):
	"""
	Remove URLs matching the patterns: 'http', 'http:', 'http:/', and 'https:/'.
	"""
	url_pattern = r'http[s]?\S+'
	return re.sub(url_pattern, '', text)


data_dir = '../data'
label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}

# MVSA-Single
data = []
with open(os.path.join(data_dir, 'MVSA_Single', 'labelResultAll.txt'), 'r') as annotations_file:
	# skip the column name
	next(annotations_file)
	
	for line in annotations_file:
		split_line = line.strip().split()

		# read the text and store it into the CSV file
		txt_path = os.path.join(data_dir, 'MVSA_Single', 'data', f'{split_line[0]}.txt')
		with open(txt_path, 'rb') as f:
			text = f.read()
			text = text.decode('ascii', 'ignore')

		(text_label, image_label) = split_line[1].split(',')
		# data.append({
		#     'ID': split_line[0], 'Text': text, 'Text label': text_label, 'Image label': image_label, 
		#     'Label': combine_text_image_label(text_label, image_label)
		# })
		data.append({'ID': split_line[0], 'Text': text, 'Label': combine_text_image_label(text_label, image_label)})

df_mvsa_single = pd.DataFrame(data)
# print(df_mvsa_single.head())
df_mvsa_single = df_mvsa_single[df_mvsa_single['Label'] != 'inconsistent']
df_mvsa_single['Text'] = df_mvsa_single['Text'].apply(remove_url)
print('MVSA-Single')
print(df_mvsa_single['Label'].value_counts())
unique_texts_count = len(df_mvsa_single['Text'].value_counts())
total_texts_count = len(df_mvsa_single)
print(f'{unique_texts_count} unique texts out of {total_texts_count} data')

df_mvsa_single['Label'] = df_mvsa_single['Label'].map(label_mapping)
# ensure that NaN values are replaced with empty strings
df_mvsa_single.fillna('', inplace=True)
df_mvsa_single.to_csv(os.path.join(data_dir, 'MVSA_Single', 'all.csv'), index=False)

# the training-test split follows CLMLF model
with open(os.path.join(data_dir, 'MVSA_Single', 'train.json'), 'r') as file:
	train_data = json.load(file)
train_ids = [item['id'] for item in train_data]
with open(os.path.join(data_dir, 'MVSA_Single', 'val.json'), 'r') as file:
	val_data = json.load(file)
val_ids = [item['id'] for item in val_data]
with open(os.path.join(data_dir, 'MVSA_Single', 'test.json'), 'r') as file:
	test_data = json.load(file)
test_ids = [item['id'] for item in test_data]

# required for creating a custom dataset in PyTorch
train_df = df_mvsa_single[df_mvsa_single['ID'].isin(train_ids)]
train_df.to_csv(os.path.join(data_dir, 'MVSA_Single', 'train.csv'), index=False)
val_df = df_mvsa_single[df_mvsa_single['ID'].isin(val_ids)]
val_df.to_csv(os.path.join(data_dir, 'MVSA_Single', 'val.csv'), index=False)
test_df = df_mvsa_single[df_mvsa_single['ID'].isin(test_ids)]
test_df.to_csv(os.path.join(data_dir, 'MVSA_Single', 'test.csv'), index=False)

# MVSA-Multiple
data = []
with open(os.path.join(data_dir, 'MVSA_Multiple', 'labelResultAll.txt'), 'r') as annotations_file:
	# skip the column name
	next(annotations_file)
	
	for line in annotations_file:
		split_line = line.strip().split()
		
		# ignore empty image files and truncated images
		image_path = os.path.join(data_dir, 'MVSA_Multiple', 'data', f'{split_line[0]}.jpg')
		try:
			image = read_image(image_path)
		except RuntimeError:
			continue

		# read the text and store it into the CSV file
		txt_path = os.path.join(data_dir, 'MVSA_Multiple', 'data', f'{split_line[0]}.txt')
		with open(txt_path, 'rb') as f:
			text = f.read()
			text = text.decode('ascii', 'ignore')

		text_label_dict = defaultdict(lambda: 0)
		image_label_dict = defaultdict(lambda: 0)
		for labels in split_line[1:]:
			(text_label, image_label) = labels.split(',')
			text_label_dict[text_label] += 1
			image_label_dict[image_label] += 1
		
		# annotated label is considered valid only when at least two of three annotators agree
		(max_label, max_count) = max(text_label_dict.items(), key=lambda a: a[1])
		text_label = max_label if max_count >= 2 else None
		(max_label, max_count) = max(image_label_dict.items(), key=lambda a: a[1])
		image_label = max_label if max_count >= 2 else None
		# data.append({
		#     'ID': split_line[0], 'Text': text, 'Text label': text_label, 'Image label': image_label, 
		#     'Label': combine_text_image_label(text_label, image_label)
		# })
		data.append({'ID': split_line[0], 'Text': text, 'Label': combine_text_image_label(text_label, image_label)})

df_mvsa_multiple = pd.DataFrame(data)
# print(df_mvsa_multiple.head())
df_mvsa_multiple = df_mvsa_multiple[~df_mvsa_multiple['Label'].isin(['invalid', 'inconsistent'])]
df_mvsa_multiple['Text'] = df_mvsa_multiple['Text'].apply(remove_url)
print('MVSA-Multiple')
print(df_mvsa_multiple['Label'].value_counts())
unique_texts_count = len(df_mvsa_multiple['Text'].value_counts())
total_texts_count = len(df_mvsa_multiple)
print(f'{unique_texts_count} unique texts out of {total_texts_count} data')

df_mvsa_multiple['Label'] = df_mvsa_multiple['Label'].map(label_mapping)
# ensure that NaN values are replaced with empty strings
df_mvsa_multiple.fillna('', inplace=True)
df_mvsa_multiple.to_csv(os.path.join(data_dir, 'MVSA_Multiple', 'all.csv'), index=False)

# the training-test split follows CLMLF model
with open(os.path.join(data_dir, 'MVSA_Multiple', 'train.json'), 'r') as file:
	train_data = json.load(file)
train_ids = [item['id'] for item in train_data]
with open(os.path.join(data_dir, 'MVSA_Multiple', 'val.json'), 'r') as file:
	val_data = json.load(file)
val_ids = [item['id'] for item in val_data]
with open(os.path.join(data_dir, 'MVSA_Multiple', 'test.json'), 'r') as file:
	test_data = json.load(file)
test_ids = [item['id'] for item in test_data]

# required for creating a custom dataset in PyTorch
train_df = df_mvsa_multiple[df_mvsa_multiple['ID'].isin(train_ids)]
train_df.to_csv(os.path.join(data_dir, 'MVSA_Multiple', 'train.csv'), index=False)
val_df = df_mvsa_multiple[df_mvsa_multiple['ID'].isin(val_ids)]
val_df.to_csv(os.path.join(data_dir, 'MVSA_Multiple', 'val.csv'), index=False)
test_df = df_mvsa_multiple[df_mvsa_multiple['ID'].isin(test_ids)]
test_df.to_csv(os.path.join(data_dir, 'MVSA_Multiple', 'test.csv'), index=False)