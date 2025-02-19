import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_and_load_model(model_name, save_dir):
    # model_name = "google/gemma-2-2b-it"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=str(Path(save_dir) / model_name),
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=str(Path(save_dir) / model_name),
        torch_dtype=torch.bfloat16,
        padding_side="left",
    )
    if 'llama' in model_name.lower():
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer