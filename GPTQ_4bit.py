import torch
from transformers import LlamaForSequenceClassification, LlamaTokenizer, AutoTokenizer, AutoModelForSequenceClassification, GPTQConfig
from datasets import load_dataset
from codecarbon import OfflineEmissionsTracker
import time
import wandb

hf_token = "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"

tracker1 = OfflineEmissionsTracker(country_iso_code="ESP", allow_multiple_runs = True, output_file= "./emissions_4bit_quant.csv", gpu_ids=[0])
tracker2 = OfflineEmissionsTracker(country_iso_code="ESP", allow_multiple_runs = True, output_file= "./emissions_4bit_eval.csv", gpu_ids=[0])
wandb.init(project="prueba_GPTQ", name="4bit_GPTQ")

# Cargar el dataset TweetEval para análisis de sentimientos
dataset_eval = load_dataset("tweet_eval", "sentiment")


# Load pre-trained Llama tokenizer and model for sequence classification
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# Load the fine-tuned model with quantization
gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)
#model = LlamaForSequenceClassification.from_pretrained("./results")
tracker1.start()
model_quantized = LlamaForSequenceClassification.from_pretrained("../bitsandbytes/results", device_map="auto", quantization_config=gptq_config)
tracker1.stop()
# Enviar ambos modelos a la GPU
#model.to("cuda")

#print(model.get_memory_footprint())
print(model_quantized.get_memory_footprint())

# Función para medir el rendimiento (precisión, tiempo y memoria)
def evaluate_model(model, tokenizer, dataset, quant):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()

    i = 0
    suma_prob = 0
    true_labels = []
    pred_labels = []
    for example in dataset["test"]:
        i += 1
        # Tokenizar el input
        inputs = tokenizer(example["text"], return_tensors="pt", padding=True, truncation=True).to("cuda")

        # Generar predicciones
        with torch.no_grad():
            outputs = model(**inputs)

        # Obtener la etiqueta predicha
        pred_label = torch.argmax(outputs.logits, dim=-1).item()
        probabilities = torch.softmax(outputs.logits, dim=-1)

	    # Obtener la probabilidad de la clase elegida
        chosen_class_probabilities = probabilities.max(dim=-1).values
        suma_prob += chosen_class_probabilities.item()
        # Comparar con la etiqueta real
        if pred_label == example["label"]:
            correct += 1
        total += 1
        pred_labels.append(pred_label)
        true_labels.append(example["label"])
        wandb.log({"Chosen_class_prob": chosen_class_probabilities})
        wandb.log({"Average_chosen_class_prob": suma_prob/i})
        if (i%1000 == 0):
            wandb.log({"Accuracy": correct/total})

    end_time = time.time()
    accuracy = correct / total
    inference_time = end_time - start_time
    # Registrar la matriz en WandB
    wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
       probs=None,
       y_true=true_labels,
       preds=pred_labels,
       class_names=["Negativo", "Neutral", "Positivo"]
    )})
    return accuracy, inference_time


#Evaular modelo sin cuantizar
#accuracy_no, time_no = evaluate_model(model, tokenizer, dataset, "not_quant")
# Evaluar el modelo cuantizado
tracker2.start()
accuracy_quantized, time_quantized = evaluate_model(model_quantized, tokenizer, dataset_eval, "quant")
tracker2.stop()
# Mostrar resultados
#print(f"Modelo no cuantizado - Precisión: {accuracy_no:.4f}, Tiempo de inferencia: {time_no:.2f}s")
print(f"Modelo cuantizado - Precisión: {accuracy_quantized:.4f}, Tiempo de inferencia: {time_quantized:.2f}s, Memoria: {model_quantized.get_memory_footprint()}")


wandb.finish() 
