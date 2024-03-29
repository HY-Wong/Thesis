import os
import chardet

# identifying text files with unknown 8-bit encoding by the following command
# find ./MVSA_Single/data -type f -name "*.txt" -exec file -I {} + | awk -F ': ' '/unknown-8bit/{print $1, $2}' > unknown_encoding_files.txt
data_dir = './../data'
count = 0

with open(os.path.join(data_dir, 'unknown_encoding_files.txt'), 'r') as file1:
	for line in file1:
		txt_file = line.strip().split()[0]
		with open(os.path.join(data_dir, txt_file), 'rb') as file2:
			text = file2.readline().strip()
			detect_result = chardet.detect(text)
			txt_file_name = os.path.basename(file2.name)
			encoding = detect_result['encoding'] if detect_result['encoding'] else 'None'
			confidence = detect_result['confidence']
			language = detect_result['language'] if detect_result['language'] else 'None'
			print(f'{txt_file_name:10} encoding: {encoding:12}, confidence: {confidence:.4f}, language: {language:12}')
		count += 1

print(f'Number of files: {count}')