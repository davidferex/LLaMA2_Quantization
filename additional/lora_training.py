import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, LlamaForSequenceClassification, LlamaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os
import time
import wandb
from codecarbon import OfflineEmissionsTracker
from huggingface_hub import login
hf_token = "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"
login(token=hf_token)

run = wandb.init(project="LLaMA_LoRA_training")


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



# Cargar el dataset TweetEval para análisis de sentimientos
dataset = load_dataset("tweet_eval", "sentiment")

# Cargar el tokenizer del modelo Llama
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

# Calcular el tamaño máximo de los tweets en tokens
max_length_train = max([len(tokenizer.tokenize(tweet)) for tweet in dataset['train']['text']])
max_length_validation = max([len(tokenizer.tokenize(tweet)) for tweet in dataset['validation']['text']])
max_length = max(max_length_train,max_length_validation)


# Función para generar tokens con el dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)


# Generación de los datasets de entrenamiento y validación
tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"].remove_columns(["text"])
validation_dataset = tokenized_datasets["validation"].remove_columns(["text"])

# Carga del modelo
model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=3, token=hf_token)
model.config.pad_token_id = model.config.eos_token_id

# Configuración LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Tarea objetivo: Sequence Classification
    r=8,                         # Dimensión low-rank (ajustable)
    lora_alpha=16,               # Factor de escalado
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # Aplicar LoRA a las capas attention
)

tracker1 = OfflineEmissionsTracker(country_iso_code="ESP", allow_multiple_runs = True, output_file= "./emissions_LoRA.csv")
tracker1.start()
# Aplicar LoRA al modelo
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Enviar modelo a la GPU
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

# Entrenamiento
training_args = TrainingArguments(
    output_dir="./results_lora",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer
)

# Entrenar el modelo
trainer.train()

# Evaluar el modelo
trainer.evaluate()

# Grabar el modelo
trainer.save_model("./results_lora")

# Grabar el modelo LoRA
model.save_pretrained("./results_lora")

# Grabar el tokenizer
tokenizer.save_pretrained("./results_lora")

tracker1.stop()
