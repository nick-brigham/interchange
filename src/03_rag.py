import pandas as pd
import requests
import chromadb
from sentence_transformers import SentenceTransformer
import time

TRAIN_FILE = 'data/processed/train.csv'
TEST_INPUT = 'data/test/test_input.csv'
TEST_ANSWERS = 'data/test/test_answers.csv'
OUTPUT_FILE = 'data/processed/rag_results.csv'

OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL = 'llama3.2:1b'
TOP_K = 5

print("Running RAG script")

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

def build_prompt(description, examples):
    example_text = '\n'.join([
        f"Customer description: {ex['description']}\nPart number: {ex['part_number']}"
        for ex in examples
    ])
    
    return f"""You are a parts interchange assistant. Use the examples below to identify the correct part number.

Examples:
{example_text}

Now identify the part number for:
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

if __name__ == '__main__':
    train_df = pd.read_csv(TRAIN_FILE)
    test_input = pd.read_csv(TEST_INPUT)
    test_answers = pd.read_csv(TEST_ANSWERS)

    print("Building store...")

    collection, embedder = build_vector_store(train_df)

    results = []
    total = len(test_input)

    print(f"\nRunning RAG inference on {total} test rows...")
    for i, row in test_input.iterrows():
        description = row['customer_description']
        examples = retrieve_similar(description, collection, embedder)
        prompt = build_prompt(description, examples)
        predicted = query_llm(prompt)
        
        results.append({
            'customer_description': description,
            'predicted': predicted
        })
        
        if len(results) % 10 == 0:
            print(f"Progress: {len(results)}/{total}")
        
        time.sleep(0.5)

    results_df = pd.DataFrame(results)
    results_df = results_df.merge(test_answers, on='customer_description', how='left')
    results_df.to_csv(OUTPUT_FILE, index=False)
    
    evaluate(results_df)