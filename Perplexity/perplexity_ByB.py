import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
from codecarbon import OfflineEmissionsTracker
import math
import os
import wandb

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Cambia si usas otra GPU

tracker1 = OfflineEmissionsTracker(country_iso_code="ESP", allow_multiple_runs = True, output_file= "./emissions_8bit_ByB_quant_perplexity.csv", gpu_ids=[1])
tracker2 = OfflineEmissionsTracker(country_iso_code="ESP", allow_multiple_runs = True, output_file= "./emissions_8bit_ByB_quant_eval_perplexity.csv", gpu_ids=[1])
wandb.init(project="Perplexity", name="8bit_ByB")

def load_llama2_model(model_id, quantization=None):
    if quantization == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantization == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.float16 if quantization else torch.float32
    )
    return tokenizer, model

def calculate_perplexity_precise(model, tokenizer, texts, max_length=1024, stride=512, device='cuda'):
    model.eval()

    nll_sum = 0.0
    n_tokens = 0

    for text in tqdm(texts, desc="Calculando perplexity"):
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.size(1)
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc

            input_ids_chunk = input_ids[:, begin_loc:end_loc]
            target_ids = input_ids_chunk.clone()
            target_ids[:, :-trg_len] = -100  # solo los últimos tokens contribuyen a la loss

            with torch.no_grad():
                outputs = model(input_ids_chunk, labels=target_ids)
                neg_log_likelihood = outputs.loss

            valid_tokens = (target_ids != -100).sum().item()
            effective_tokens = valid_tokens - target_ids.size(0)  # ajustar por el shift interno
            nll_sum += neg_log_likelihood.item() * effective_tokens
            n_tokens += effective_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    avg_nll = nll_sum / n_tokens
    ppl = math.exp(avg_nll)
    wandb.log({"Perplexity": ppl})
    return ppl

if __name__ == "__main__":
    model_id = "meta-llama/Llama-2-7b-hf"  # Asegúrate de tener acceso
    quantization = "8bit"  # Cambia a "8bit" o "4bit" si lo deseas

    print(f"Cargando modelo {model_id} con cuantización: {quantization or 'sin cuantizar'}")
    tracker1.start()
    tokenizer, model = load_llama2_model(model_id, quantization)
    tracker1.stop()

    print("Cargando WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:100%]")
    texts = [sample["text"] for sample in dataset if sample["text"].strip()]

    print("Calculando perplexity (preciso)...")
    tracker2.start()
    ppl = calculate_perplexity_precise(model, tokenizer, texts)
    tracker2.stop()
    print(f"\n✅ Perplexity precisa ({quantization or 'no quant'}): {ppl:.2f}")
