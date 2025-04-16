import os
# Las GPUs se identifican como con nvidia-smi
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# La GPU visible al script es sólo la primera
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from transformers import LlamaTokenizer, LlamaForSequenceClassification, LlamaConfig, GPTQConfig
from datasets import load_dataset
import time

from codecarbon import OfflineEmissionsTracker
import wandb

# Para imprimir la matriz de confusión
from sklearn.metrics import classification_report

# Barra de progreso de la inferencia
from torch.utils.data import DataLoader
from tqdm import tqdm 

tracker1 = OfflineEmissionsTracker(country_iso_code="ESP", allow_multiple_runs = True, output_file= "./emissions_4bit_quant_batch64_GPTQ_LORA.csv", gpu_ids=[0])
tracker2 = OfflineEmissionsTracker(country_iso_code="ESP", allow_multiple_runs = True, output_file= "./emissions_4bit_eval_batch64_GPTQ_LORA.csv", gpu_ids=[0])

wandb.init(project="GPTQ_Lora", name="4bit_batch64")

# Imprimo el nombre de GPU's disponibles para el script
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Cargar del dataset TweetEval para análisis de sentimientos
dataset = load_dataset("tweet_eval", "sentiment")

# Nombre del modelo
model_name = '../bitsandbytes/results_lora'

# Clave Huggingface
hf_token = "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"

# Dispositivo: solo se indica cuda, en lugar de cuda:0, ya que sólo está visible el 0
device = 'cuda'

# Cargar el tokenizador
tokenizer = LlamaTokenizer.from_pretrained(
    model_name,
    token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

tracker1.start()
# Cargar el modelo, indicando con LlamaConfig la clasificación (por LoRA)
config = LlamaConfig.from_pretrained(model_name, num_labels=3)
gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
model = LlamaForSequenceClassification.from_pretrained(
    model_name,
    token=hf_token,
    config=config,
    device_map="auto",
    quantization_config=gptq_config)
model.config.pad_token_id = model.config.eos_token_id
tracker1.stop()

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
    max_memory_usage = 0

    # Configura el dataset para que devuelva tensores de PyTorch, para usar DataLoader de PyTorch
    dataset.set_format("torch")
    test_dataset = dataset["test"]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Calculamos el tamaño máximo de tokens
    max_length = max([len(tokenizer.tokenize(tweet)) for tweet in test_dataset["text"]])

    for batch in tqdm(test_loader, desc="Test"):
        # Sincronizar y medir la memoria antes
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Tokenizar el batch
        inputs = tokenizer(batch['text'], return_tensors="pt",
                           padding="max_length", truncation=True, max_length=max_length).to(device)

        # Inferencia. Desactivar el cálculo de gradientes
        with torch.no_grad():
            outputs = model(**inputs)
            pred_batch = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

        # Medir memoria utilizada despu  s de la inferencia
        memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convertir a MB
        max_memory_usage = max(max_memory_usage, memory_used)

        true_labels.extend(batch['label'])
        pred_labels.extend(pred_batch)

    end_time = time.time()
    inference_time = end_time - start_time

    # Imprimir la matriz de confiusión
    print(classification_report(true_labels, pred_labels, target_names=["Negative", "Neutral", "Positive"]))

    wandb.log({"Max_memory_usage_MB": max_memory_usage})

    return inference_time

tracker2.start()
# Evaluar modelo
time_no = evaluate_model(model, tokenizer, dataset, 64)
tracker2.stop()

# Mostrar resultados
print(f"Modelo LoRA cuantizado gptq 8bit - Tiempo de inferencia: {time_no:.2f}s")