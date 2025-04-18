import os
from datasets import load_dataset
import time
import re
from sklearn.metrics import classification_report

# Especificar GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
from transformers import pipeline

# Nombre del modelo en Hugging Face
model_name = "meta-llama/Llama-2-7b-hf"
 
hf_token = "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"

generator = pipeline("text-generation", model=model_name, device=0, token=hf_token)

# Cargar el dataset TweetEval para análisis de sentimientos
dataset = load_dataset("tweet_eval", "sentiment")

correct = 0
total = 0
invalid = 0
start_time = time.time()

true_labels = []
predicted_labels = []

label_code = ['negative', 'neutral', 'positive']
er_label = re.compile(r'(?i)(negative)|(neutral)|(positive)')

for sample in dataset["test"]:
    tweet = sample["text"]
    prompt = f'Analyze the sentiment of the following tweet. ' \
             f'Respond with ONLY one word: "positive", "negative", or "neutral". Do not generate anything else.\n\n' \
             f'Tweet: "{tweet}"\n' \
             f'Sentiment:'

    result = generator(prompt, max_new_tokens=5, truncation=True, return_full_text=False)

    # Obtener la etiqueta predicha
    m = er_label.search(result[0]['generated_text'])
    if m:
        predicted_label = label_code[m.lastindex-1]
        true_label = label_code[sample["label"]]

        # Comparar con la etiqueta real
        if predicted_label == true_label:
            correct += 1
        total += 1
        predicted_labels.append(predicted_label)
        true_labels.append(true_label)

        print(classification_report(
            true_labels, predicted_labels,
            labels=["negative", "neutral", "positive"]
        ))
    else:
        invalid += 1

