import os
# Las GPUs se identifican como con nvidia-smi
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# La GPU visible al script es sólo la primera
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from transformers import LlamaTokenizer, LlamaForSequenceClassification, LlamaConfig
from datasets import load_dataset
import time
# Para imprimir la matriz de confusión
from sklearn.metrics import classification_report
# Barra de progreso de la inferencia
from torch.utils.data import DataLoader
from tqdm import tqdm

# Imprimo el nombre de GPU's disponibles para el script
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Cargar del dataset TweetEval para análisis de sentimientos
dataset = load_dataset("tweet_eval", "sentiment")

# Nombre del modelo
model_name = './results_lora'

# Clave Huggingface
hf_token = "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"

# Dispositivo: solo se indica cuda, en lugar de cuda:0, ya que sólo está visible el 0
device = 'cuda'

# Cargar el tokenizador
tokenizer = LlamaTokenizer.from_pretrained(
    model_name,
    token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

# Cargar el modelo, indicando con LlamaConfig la clasificación (por LoRA)
config = LlamaConfig.from_pretrained(model_name, num_labels=3)
model = LlamaForSequenceClassification.from_pretrained(
    model_name,
    token=hf_token,
    config=config).to(device)
model.config.pad_token_id = model.config.eos_token_id

# Enviar modelo a la GPU en una espera activa hasta que haya memoria
loaded = False
num_tries = 0
while not loaded:
    try:
        model.to("cuda")
        loaded = True
    except torch.OutOfMemoryError:
        num_tries += 1
        print('New try:', num_tries)
        time.sleep(5)


def evaluate_model(model, tokenizer, dataset, batch_size=8):
    # Poner el modelo en modo de evaluación
    model.eval()

    start_time = time.time()
    true_labels = []
    pred_labels = []

    # Configura el dataset para que devuelva tensores de PyTorch, para usar DataLoader de PyTorch
    dataset.set_format("torch")
    test_dataset = dataset["test"]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Calculamos el tamaño máximo de tokens
    max_length = max([len(tokenizer.tokenize(tweet)) for tweet in test_dataset["text"]])

    for batch in tqdm(test_loader, desc="Test"):
        # Tokenizar el batch
        inputs = tokenizer(batch['text'], return_tensors="pt",
                           padding="max_length", truncation=True, max_length=max_length).to(device)

        # Inferencia. Desactivar el cálculo de gradientes
        with torch.no_grad():
            outputs = model(**inputs)
            pred_batch = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

        true_labels.extend(batch['label'])
        pred_labels.extend(pred_batch)

    end_time = time.time()
    inference_time = end_time - start_time

    # Imprimir la matriz de confiusión
    print(classification_report(true_labels, pred_labels, target_names=["Negative", "Neutral", "Positive"]))

    return inference_time


# Evaluar modelo LoRA sin cuantizar
time_no = evaluate_model(model, tokenizer, dataset, 32)

# Mostrar resultados
print(f"Modelo LoRA no cuantizado - Tiempo de inferencia: {time_no:.2f}s")
