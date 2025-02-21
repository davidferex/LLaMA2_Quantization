import torch
from transformers import LlamaForSequenceClassification, LlamaTokenizer, AutoTokenizer, AutoModelForSequenceClassification, QuantoConfig
from datasets import load_dataset
from codecarbon import OfflineEmissionsTracker
from optimum.quanto import QuantizedModelForCausalLM, qint4, qint8
import time
import wandb



hf_token = "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"

tracker1 = OfflineEmissionsTracker(country_iso_code="ESP", allow_multiple_runs = True, output_file= "./emissions_8bit_batch32_quant_Quanto.csv", gpu_ids=[0])
tracker2 = OfflineEmissionsTracker(country_iso_code="ESP", allow_multiple_runs = True, output_file= "./emissions_8bit_batch32_eval_Quanto.csv", gpu_ids=[0])
wandb.init(project="Quanto", name="8bit_batch32")

# Cargar el dataset TweetEval para análisis de sentimientos
dataset_eval = load_dataset("tweet_eval", "sentiment")


# Load pre-trained Llama tokenizer and model for sequence classification
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
tokenizer.pad_token = tokenizer.eos_token
# Load the fine-tuned model with quantization
#quant_config = QuantoConfig(weights="int8")
#model = LlamaForSequenceClassification.from_pretrained("./results")
model = LlamaForSequenceClassification.from_pretrained("../bitsandbytes/results")
tracker1.start()
#model_quantized = LlamaForSequenceClassification.from_pretrained("../bitsandbytes/results", device_map="auto", quantization_config=quant_config, use_auth_token=hf_token)
model_quantized = QuantizedModelForCausalLM.quantize(model, weights=qint8, exclude='lm_head')
tracker1.stop()
# Enviar ambos modelos a la GPU
#model.to("cuda")


# Función para medir el rendimiento (precisión, tiempo y memoria)
def evaluate_model(model, tokenizer, dataset, quant, batch_size=32):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()

    suma_prob = 0
    true_labels = []
    pred_labels = []
    max_memory_usage = 0

    for i in range(0, len(dataset["test"]["text"]), batch_size):
        texts = dataset["test"]["text"][i:i+batch_size]
        labels = dataset["test"]["label"][i:i+batch_size]

        # Tokenizar el input en batch
        inputs = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True).to("cuda")

        # Sincronizar y medir la memoria antes
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Generar predicciones
        with torch.no_grad():
            outputs = model(**inputs)

        # Medir memoria utilizada después de la inferencia
        memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convertir a MB
        max_memory_usage = max(max_memory_usage, memory_used)
        

        # Obtener las etiquetas predichas
        pred_label_batch = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
        probabilities = torch.softmax(outputs.logits, dim=-1)
        chosen_class_probabilities = probabilities.max(dim=-1).values.cpu().tolist()
        suma_prob += sum(chosen_class_probabilities)

        # Comparar con las etiquetas reales
        correct += sum([1 for pred, true in zip(pred_label_batch, labels) if pred == true])
        total += len(labels)
        pred_labels.extend(pred_label_batch)
        true_labels.extend(labels)
        wandb.log({"Chosen_class_prob": sum(chosen_class_probabilities)/len(chosen_class_probabilities),
                    "Average_chosen_class_prob": suma_prob/total,
                    "Accuracy": correct/total,
                    "Memory_used_MB": memory_used})

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
    wandb.log({"Max_memory_usage_MB": max_memory_usage})
    return accuracy, inference_time, max_memory_usage


#Evaular modelo sin cuantizar
#accuracy_no, time_no = evaluate_model(model, tokenizer, dataset, "not_quant")
# Evaluar el modelo cuantizado
tracker2.start()
accuracy_quantized, time_quantized, max_memory = evaluate_model(model_quantized, tokenizer, dataset_eval, "quant")
tracker2.stop()
# Mostrar resultados
#print(f"Modelo no cuantizado - Precisión: {accuracy_no:.4f}, Tiempo de inferencia: {time_no:.2f}s")
print(f"Modelo cuantizado - Precisión: {accuracy_quantized:.4f}, Tiempo de inferencia: {time_quantized:.2f}s, Memoria: {max_memory}")


wandb.finish() 
