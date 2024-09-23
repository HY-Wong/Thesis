import os
import torch
import transformers
import pandas as pd
import nltk
import re

from textblob import TextBlob, Word


###
access_token = ''
model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=access_token)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

pipeline = transformers.pipeline(
    'text-generation',
    model=model_id,
    model_kwargs={'torch_dtype': torch.bfloat16},
    device=device,
    # token=access_token,
    pad_token_id=tokenizer.eos_token_id
)


def extract_objects_from_text_llm(text):
    messages = [
        {"role": "system", "content": "Your goal is to extract all the concrete nouns. You will only return the concrete nouns in lower case and separate them by commas without any additional comments."},
        {"role": "user", "content": f"Extract all concrete nouns mentioned in this sentence: {text}"},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=64,
    )
    return outputs[0]['generated_text'][-1]['content']


def post_process(row):
    object_list = row['Objects_LLM'].split(',')
    object_list = [object.strip() for object in object_list]
    # filter out objects that are not in the original text
    object_list = [object for object in object_list if object != '' and Word(object).singularize() in row['Text'].lower()]
    return object_list


def extract_objects_from_text_textblob(text):
    # remove '@' and '#'
    text = re.sub(r'[@#]', '', text)
    blob = TextBlob(text)
    object_list = [word for word, pos in blob.tags if pos in ('NN', 'NNS')] 
    return object_list


dataset_path = '../data/MVSA_Single'
df = pd.read_csv(os.path.join(dataset_path, 'all.csv'), keep_default_na=False)

df['Objects_LLM'] = df['Text'].apply(extract_objects_from_text_llm)
df['Objects_LLM'] = df.apply(post_process, axis=1)
df['Objects_TextBlob'] = df['Text'].apply(extract_objects_from_text_textblob)
df.to_csv(os.path.join(dataset_path, 'all.csv'), index=False)

dataset_path = '../data/MVSA_Multiple'
df = pd.read_csv(os.path.join(dataset_path, 'all.csv'), keep_default_na=False)

df['Objects_LLM'] = df['Text'].apply(extract_objects_from_text_llm)
df['Objects_LLM'] = df.apply(post_process, axis=1)
df['Objects_TextBlob'] = df['Text'].apply(extract_objects_from_text_textblob)
df.to_csv(os.path.join(dataset_path, 'all.csv'), index=False)

dataset_path = '../data/Example'
df = pd.read_csv(os.path.join(dataset_path, 'example.csv'), keep_default_na=False)

df['Objects_LLM'] = df['Text'].apply(extract_objects_from_text_llm)
df['Objects_LLM'] = df.apply(post_process, axis=1)
df['Objects_TextBlob'] = df['Text'].apply(extract_objects_from_text_textblob)
df.to_csv(os.path.join(dataset_path, 'example.csv'), index=False)