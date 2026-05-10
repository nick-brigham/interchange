import os
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

TRAIN_FILE = 'data/processed/train.csv'
MODEL_NAME = 'ollama_models/llama3.2:1b'
OUTPUT_DIR = 'models/finetuned'
OLLAMA_MODEL = 'llama3.2:1b'

def format_row(row):
    return {
        'text': f"""You are a parts interchange assistant. Identify the correct part number.

Customer description: {row['customer_description']}
Part number: {row['part_number']}"""
    }

def load_and_format_data(filepath):
    df = pd.read_csv(filepath)
    formatted = df.apply(format_row, axis=1).tolist()
    dataset = Dataset.from_list(formatted)
    print(f"Loaded {len(dataset)} training examples")
    return dataset

def load_model_and_tokenizer():
    print("Loading model and tokenizer...")
    model_id = "meta-llama/Llama-3.2-1B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map='cpu'
    )
    return model, tokenizer

def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj']
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def train(model, tokenizer, dataset):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to='none'
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

print("running script")

if __name__ == '__main__':
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    dataset = load_and_format_data(TRAIN_FILE)
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
    train(model, tokenizer, dataset)