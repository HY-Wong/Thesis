import os
import ast
import torch
import matplotlib.pyplot as plt
import pandas as pd

from typing import Dict, Any
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


data_dir = '../data'
image_dir = '../image'

plt.rcParams['font.family'] = 'serif'


def plot_detection(
	image_path: str, detection: Dict[str, Any], file_path: str, file_name: str, num_regions: int = None
):
	"""
	Visualize bounding boxes and detected labels on the image.
	"""
	image = Image.open(image_path)
	
	boxes = detection['boxes']
	labels = detection['labels']
	scores = detection['scores']

	# select the objects with the highest m confidence scores
	if num_regions != None:
		boxes = boxes[:num_regions]
		labels = labels[:num_regions]
		scores = scores[:num_regions]

	pred_labels = []
	for i in range(boxes.shape[0]):
		s = f'{labels[i]}: {scores[i]:.2f}'
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

	plt.savefig(os.path.join(file_path, f'{file_name}.png'), dpi=480)
	plt.close()


def grounded_object_detection(text: str, image, score_threshold: float = 0.1, area_threshold: float = 0.8):
	"""
	Extract detected objects and their corresponding region features from Grounding DINO, 
	which are used as affective regions.
	"""
	inputs = processor(images=image, text=text, return_tensors='pt').to(device)
	with torch.no_grad():
		outputs = model(**inputs)

	# print(outputs.logits.shape)
	# print(outputs.pred_boxes.shape)
	# print(outputs.last_hidden_state.shape)

	# post-process the grounded object detection results without filtering any boxes,
	# ensuring alignment between region features and detected objects.
	results = processor.post_process_grounded_object_detection(
		outputs,
		inputs.input_ids,
		box_threshold=0.0,
		text_threshold=0.1,
		target_sizes=[image.size[::-1]]
	)
	detection = results[0]
	assert(detection['scores'].shape[0] == 900)

	# define the output of the decoder as the region features
	boxes = detection['boxes']
	region_features = outputs.last_hidden_state.squeeze()
	labels = detection['labels']
	scores = detection['scores']

	# ensure sufficient detection quality
	mask = scores >= score_threshold
	boxes = boxes[mask]
	region_features = region_features[mask]
	labels = [item for item, m in zip(labels, mask.tolist()) if m]
	scores = scores[mask]

	# filter out boxes that are too large, as some detection results may cover 
	# the entire original image
	h, w = image.size[::-1]
	max_area = h * w * area_threshold
	# (x0, y0, x1, y1)
	box_areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
	mask = box_areas <= max_area
	boxes = boxes[mask]
	region_features = region_features[mask]
	labels = [item for item, m in zip(labels, mask.tolist()) if m]
	scores = scores[mask]

	min_boxes = 6
	max_boxes = 12
	sorted_indices = torch.argsort(scores, descending=True)
	sorted_indices = sorted_indices[:max_boxes]
	boxes = boxes[sorted_indices]
	region_features = region_features[sorted_indices]
	labels = [labels[index] for index in sorted_indices.tolist()]
	scores = scores[sorted_indices]

	# if there are not enough detected objects above the threshold, pad with zeros
	num_regions = boxes.shape[0]
	if num_regions < min_boxes:
		padding_num = min_boxes - num_regions
		boxes = torch.cat((boxes, torch.zeros(padding_num, 4).to(device)), dim=0)
		region_features = torch.cat((region_features, torch.zeros(padding_num, 256).to(device)), dim=0)
		labels.extend([''] * padding_num)
		scores = torch.cat((scores, torch.zeros(padding_num).to(device)), dim=0)

	# print(boxes.shape)
	# print(region_features.shape)
	# print(len(labels))
	# print(scores.shape)

	detections_dict = {
		'boxes': boxes.to('cpu'),
		'region_features': region_features.to('cpu'),
		'labels': labels,
		'scores': scores.to('cpu')
	}
	return detections_dict


# Grounding DINO model
model_id = 'IDEA-Research/grounding-dino-tiny'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
# print(model)

dataset_path = os.path.join(data_dir, 'MVSA_Multiple')
df = pd.read_csv(os.path.join(dataset_path, 'all.csv'), keep_default_na=False)
# convert the string representation of lists to actual lists
df['Objects_LLM'] = df['Objects_LLM'].apply(ast.literal_eval)
df['Objects_TextBlob'] = df['Objects_TextBlob'].apply(ast.literal_eval)
# remove duplicates within each list
df['Objects_LLM'] = df['Objects_LLM'].apply(lambda x: list(set(x)))
df['Objects_TextBlob'] = df['Objects_TextBlob'].apply(lambda x: list(set(x)))

self_defined_objects = ['person', 'face', 'animal', 'tree', 'flower']

# create separate directories to store each feature
for i in range(1, 6):
	os.makedirs(os.path.join(dataset_path, f'features_grounding_dino{i}'), exist_ok=True)

# five text prompts for open-set detection for each image in the dataset
for _, row in df.iterrows():
	text1 = 'everything.'
	text2 = 'everything.' + ''.join([object + '.' for object in row['Objects_TextBlob']]) 
	text3 = 'everything.' + ''.join([object + '.' for object in row['Objects_LLM']]) 
	text4 = 'everything.' + ''.join([object + '.' for object in self_defined_objects])
	combined_objects = list(set(row['Objects_LLM'] + self_defined_objects))
	text5 = 'everything.' + ''.join([object + '.' for object in combined_objects])
	text_list = [text1, text2, text3, text4, text5]
	image_path = os.path.join(dataset_path, 'data', f"{row['ID']}.jpg")
	image = Image.open(image_path)

	for i, text in enumerate(text_list):
		detection = grounded_object_detection(text, image)
		# plot_detection(image_path, detection, image_dir, f'a{_:02d}_{i:02}', num_regions=4)
		torch.save(detection, os.path.join(dataset_path, f'features_grounding_dino{i+1}', f"{row['ID']}.pth"))