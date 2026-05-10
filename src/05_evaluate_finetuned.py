import os
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time

TEST_INPUT = 'data/test/test_input.csv'
TEST_ANSWERS = 'data/test/test_answers.csv'
OUTPUT_FILE = 'data/processed/finetuned_results.csv'
BASE_MODEL = 'meta-llama/Llama-3.2-1B'
FINETUNED_DIR = 'models/finetuned'

def load_model():
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map='cpu'
    )
    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(base_model, FINETUNED_DIR)
    model.eval()
    return model, tokenizer

def generate_part_number(description, model, tokenizer):
    prompt = f"""You are a parts interchange assistant. Identify the correct part number.

Customer description: {description}
Part number:"""

    inputs = tokenizer(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only what comes after "Part number:"
    if 'Part number:' in full_output:
        answer = full_output.split('Part number:')[-1].strip()
        # Take only the first line
        answer = answer.split('\n')[0].strip()
    else:
        answer = full_output.strip()
    
    return answer

def evaluate(results_df):
    correct = (results_df['predicted'].str.upper() == results_df['part_number'].str.upper()).sum()
    total = len(results_df)
    accuracy = correct / total * 100
    print(f"\nResults:")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    return accuracy

if __name__ == '__main__':
    test_input = pd.read_csv(TEST_INPUT)
    test_answers = pd.read_csv(TEST_ANSWERS)
    
    model, tokenizer = load_model()
    
    results = []
    total = len(test_input)
    
    print(f"\nRunning inference on {total} test rows...")
    for i, row in test_input.iterrows():
        description = row['customer_description']
        predicted = generate_part_number(description, model, tokenizer)
        results.append({
            'customer_description': description,
            'predicted': predicted
        })
        
        if len(results) % 10 == 0:
            print(f"Progress: {len(results)}/{total}")
        
        time.sleep(0.1)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.merge(test_answers, on='customer_description', how='left')
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    evaluate(results_df)