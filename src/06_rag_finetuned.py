import os
import sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import torch
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import time

TRAIN_FILE = 'data/processed/train.csv'
TEST_INPUT = 'data/test/test_input.csv'
TEST_ANSWERS = 'data/test/test_answers.csv'
OUTPUT_FILE = 'data/processed/rag_finetuned_results.csv'
BASE_MODEL = 'meta-llama/Llama-3.2-1B'
FINETUNED_DIR = 'models/finetuned'
TOP_K = 5

def load_model():
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map='cpu'
    )
    model = PeftModel.from_pretrained(base_model, FINETUNED_DIR)
    model.eval()
    return model, tokenizer

def build_vector_store(train_df):
    print("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Building vector store...")
    client = chromadb.Client()
    collection = client.create_collection("parts")
    
    descriptions = train_df['customer_description'].tolist()
    part_numbers = train_df['part_number'].tolist()
    
    embeddings = embedder.encode(descriptions, show_progress_bar=True)
    
    collection.add(
        embeddings=embeddings.tolist(),
        documents=descriptions,
        metadatas=[{'part_number': pn} for pn in part_numbers],
        ids=[str(i) for i in range(len(descriptions))]
    )
    
    print(f"Vector store built with {len(descriptions)} entries")
    return collection, embedder

def retrieve_similar(description, collection, embedder, top_k=TOP_K):
    embedding = embedder.encode([description]).tolist()
    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k
    )
    
    examples = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        examples.append({
            'description': doc,
            'part_number': meta['part_number']
        })
    return examples

def generate_part_number(description, examples, model, tokenizer):
    example_text = '\n'.join([
        f"Customer description: {ex['description']}\nPart number: {ex['part_number']}"
        for ex in examples
    ])
    
    prompt = f"""You are a parts interchange assistant. Use the examples below to identify the correct part number.

Examples:
{example_text}

Now identify the part number for:
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
    if 'Part number:' in full_output:
        answer = full_output.split('Part number:')[-1].strip()
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
    train_df = pd.read_csv(TRAIN_FILE)
    test_input = pd.read_csv(TEST_INPUT)
    test_answers = pd.read_csv(TEST_ANSWERS)
    
    model, tokenizer = load_model()
    collection, embedder = build_vector_store(train_df)
    
    results = []
    total = len(test_input)
    
    print(f"\nRunning RAG + Fine-tuned inference on {total} test rows...")
    for i, row in test_input.iterrows():
        description = row['customer_description']
        examples = retrieve_similar(description, collection, embedder)
        predicted = generate_part_number(description, examples, model, tokenizer)
        
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