import torch
from transformers import LlamaForSequenceClassification, LlamaConfig, LlamaTokenizer, AutoTokenizer, AutoModelForSequenceClassification, QuantoConfig, BitsAndBytesConfig, GPTQConfig, HqqConfig
from datasets import load_dataset
from codecarbon import OfflineEmissionsTracker
from optimum.quanto import QuantizedModelForCausalLM, qint4, qint8
import time
import wandb


# Load pre-trained Llama tokenizer and model for sequence classification
model_name = '../bitsandbytes/results_lora'

hf_token = "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"


tokenizer = LlamaTokenizer.from_pretrained(
    model_name,
    token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

config = LlamaConfig.from_pretrained(model_name, num_labels=3)
model = LlamaForSequenceClassification.from_pretrained(
    model_name,
    token=hf_token,
    config=config,
    device_map="auto")
model.config.pad_token_id = model.config.eos_token_id
model = QuantizedModelForCausalLM.quantize(model, weights=qint8, exclude='lm_head')
model.save_pretrained('./model_quantized_Quanto8b')

