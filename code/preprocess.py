import numpy as np
import pandas as pd
import os
import re
import subprocess

from collections import defaultdict
from typing import Union
from torchvision.io import read_image


def convert_to_utf8(file_path: str, from_encoding: Union[str, None] = None) -> str:
	if from_encoding:
		result = subprocess.run(
			['iconv', '-f', from_encoding, '-t', 'UTF-8', '-c', file_path], 
			stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
		)
	else:
		result = subprocess.run(
			['iconv', '-t', 'UTF-8', '-c', file_path], 
			stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
		)
	return result.stdout.strip()


def combine_text_image_label(text_label: Union[str, None], image_label: Union[str, None]) -> str:
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
	# remove URLs, 'http', 'http:', 'http:/', and 'https:/'
	url_pattern = r'http[s]?\S+'
	return re.sub(url_pattern, '', text)


data_dir = './../data'
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
		text = convert_to_utf8(txt_path, 'GB2312')

		(text_label, image_label) = split_line[1].split(',')
		# data.append({
		#     'ID': split_line[0], 'Text': text, 'Text label': text_label, 'Image label': image_label, 
		#     'Label': combine_text_image_label(text_label, image_label)
		# })
		data.append({'ID': split_line[0], 'Text': text, 'Label': combine_text_image_label(text_label, image_label)})

df_mvsa_single = pd.DataFrame(data)
# print(df_mvsa_single.head())
df_mvsa_single = df_mvsa_single[df_mvsa_single['Label'] != 'inconsistent']
print('MVSA-Single')
print(df_mvsa_single['Label'].value_counts())
unique_texts_count = len(df_mvsa_single['Text'].value_counts())
total_texts_count = len(df_mvsa_single)
print(f'{unique_texts_count} unique texts out of {total_texts_count} data')

# required for creating a custom dataset in PyTorch
df_mvsa_single['Label'] = df_mvsa_single['Label'].map(label_mapping)
df_mvsa_single['Text'] = df_mvsa_single['Text'].apply(remove_url)
# ensure that NaN values are replaced with empty strings
df_mvsa_single.fillna('', inplace=True)
df_mvsa_single.to_csv(os.path.join(data_dir, 'MVSA_Single', 'MVSA_Single.csv'), index=False)

# MVSA-Multiple
data = []
with open(os.path.join(data_dir, 'MVSA_Multiple', 'labelResultAll.txt'), 'r') as annotations_file:
	# skip the column name
	next(annotations_file)
	
	for line in annotations_file:
		split_line = line.strip().split()

		# ignore empty image files and truncated images
		image_path = os.path.join(data_dir, 'MVSA_Multiple', 'data', '{}.jpg'.format(split_line[0]))
		try:
			image = read_image(image_path)
		except RuntimeError:
			continue

		# read the text and store it into the CSV file
		txt_path = os.path.join(data_dir, 'MVSA_Multiple', 'data', f'{split_line[0]}.txt')
		text = convert_to_utf8(txt_path)

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
print('MVSA-Multiple')
print(df_mvsa_multiple['Label'].value_counts())
unique_texts_count = len(df_mvsa_multiple['Text'].value_counts())
total_texts_count = len(df_mvsa_multiple)
print(f'{unique_texts_count} unique texts out of {total_texts_count} data')

# required for creating a custom dataset in PyTorch
df_mvsa_multiple['Label'] = df_mvsa_multiple['Label'].map(label_mapping)
df_mvsa_multiple['Text'] = df_mvsa_multiple['Text'].apply(remove_url)
# ensure that NaN values are replaced with empty strings
df_mvsa_multiple.fillna('', inplace=True)
df_mvsa_multiple.to_csv(os.path.join(data_dir, 'MVSA_Multiple', 'MVSA_Multiple.csv'), index=False)