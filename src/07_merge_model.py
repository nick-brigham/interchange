import os
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = 'meta-llama/Llama-3.2-1B'
FINETUNED_DIR = 'models/finetuned'
MERGED_DIR = 'models/merged'

if __name__ == '__main__':
    os.makedirs(MERGED_DIR, exist_ok=True)
    
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map='cuda',
        local_files_only=True
    )
    print("Base model loaded.")

    print("Loading LoRA weights...")
    try:
        model = PeftModel.from_pretrained(base_model, FINETUNED_DIR)
        print("LoRA loaded.")
    except Exception as e:
        print(f"ERROR loading LoRA: {e}")
        sys.exit(1)

    print("Merging weights...")
    try:
        model = model.merge_and_unload()
        print("Merge complete.")
    except Exception as e:
        print(f"ERROR merging: {e}")
        sys.exit(1)

    print("Saving merged model...")
    try:
        model.save_pretrained(MERGED_DIR)
        tokenizer.save_pretrained(MERGED_DIR)
        print(f"Merged model saved to {MERGED_DIR}")
    except Exception as e:
        print(f"ERROR saving: {e}")
        sys.exit(1)