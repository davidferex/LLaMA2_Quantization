import torch
from transformers import LlamaForSequenceClassification, LlamaTokenizer, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from datasets import load_dataset
from codecarbon import OfflineEmissionsTracker
import time
import wandb

hf_token = "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"

tracker1 = OfflineEmissionsTracker(country_iso_code="ESP", allow_multiple_runs = True, output_file= "./emissions_not_quant_eval2.csv")
wandb.init(project="All_together_ByB_2", name="not_quant")

# Cargar el dataset TweetEval para análisis de sentimientos
dataset = load_dataset("tweet_eval", "sentiment")

# Load pre-trained Llama tokenizer and model for sequence classification
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# Load the fine-tuned model with quantization
#bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = LlamaForSequenceClassification.from_pretrained("./results")
#model_quantized = LlamaForSequenceClassification.from_pretrained("./results", quantization_config=bnb_config)

# Enviar ambos modelos a la GPU
model.to("cuda")

print(model.get_memory_footprint())
#print(model_quantized.get_memory_footprint())

# Función para medir el rendimiento (precisión, tiempo y memoria)
def evaluate_model(model, tokenizer, dataset, quant):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()

    suma_prob = 0
    true_labels = []
    pred_labels = []
    for example in dataset["test"]:
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
        wandb.log({"Average_chosen_class_prob": suma_prob/total})
        if (total%1000 == 0):
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
tracker1.start()
accuracy_no, time_no = evaluate_model(model, tokenizer, dataset, "not_quant")
tracker1.stop()
# Evaluar el modelo cuantizado
#accuracy_quantized, time_quantized = evaluate_model(model_quantized, tokenizer, dataset, "quant")

# Mostrar resultados
print(f"Modelo no cuantizado - Precisión: {accuracy_no:.4f}, Tiempo de inferencia: {time_no:.2f}s")
#print(f"Modelo cuantizado - Precisión: {accuracy_quantized:.4f}, Tiempo de inferencia: {time_quantized:.2f}s")


wandb.finish() 
