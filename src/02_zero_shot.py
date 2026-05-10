import pandas as pd
import requests
import json
import time

INPUT_FILE = 'data/test/test_input.csv'
ANSWERS_FILE = 'data/test/test_answers.csv'
OUTPUT_FILE = 'data/processed/zero_shot_results.csv'

OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL = 'llama3.2:1b'

def build_prompt(description):
    return f"""You are a parts interchange assistant. Your job is to identify the correct manufacturer part number from a customer description.

Customer description: {description}

Respond with ONLY the part number, nothing else. No explanation, no punctuation, just the part number."""

def query_llm(prompt):
    payload = {
        'model': MODEL,
        'prompt': prompt,
        'stream': False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    return response.json()['response'].strip()

def evaluate(results_df):
    correct = (results_df['predicted'].str.upper() == results_df['part_number'].str.upper()).sum()
    total = len(results_df)
    accuracy = correct / total * 100
    print(f"\nResults:")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    return accuracy

print("Running script...")

if __name__ == '__main__':
    test_input = pd.read_csv(INPUT_FILE)
    test_answers = pd.read_csv(ANSWERS_FILE)

    results = []
    total = len(test_input)

    for i, row in test_input.iterrows():
        prompt = build_prompt(row['customer_description'])
        predicted = query_llm(prompt)
        results.append({
            'customer_description': row['customer_description'],
            'predicted': predicted
        })
        # Progress update every 10 rows
        if (len(results)) % 10 == 0:
            print(f"Progress: {len(results)}/{total}")
        time.sleep(0.5)

    results_df = pd.DataFrame(results)
    results_df = results_df.merge(test_answers, on='customer_description', how='left')
    results_df.to_csv(OUTPUT_FILE, index=False)

    evaluate(results_df)