# Part Interchange Assistant

A local LLM-powered application that identifies manufacturer part numbers from customer descriptions. Built as a portfolio project to demonstrate data pipeline development, LLM fine-tuning, and containerized deployment.

## Project Overview

Given a customer-supplied product description such as `5VX840 SUPER HC V-BELT`, the application returns the correct manufacturer part number (`5VX840`). This is a true interchange problem — the mapping between customer descriptions and part numbers is not always obvious and cannot be solved by simple text extraction.

## Approach

Four LLM strategies were tested and compared:

| Approach | Accuracy | Notes |
|---|---|---|
| Zero-shot prompting | 20% | No domain knowledge, high hallucination rate |
| RAG (Retrieval Augmented Generation) | 51.5% | Significant improvement via catalog retrieval |
| Fine-tuned (LoRA) | 96% | Best performer, learned mappings directly |
| RAG + Fine-tuned | 93.6% | Retrieved examples introduced conflicting signals |

The fine-tuned model was selected for the production application.

## Tech Stack

- **LLM** — Llama 3.2 1B (Meta), fine-tuned with LoRA via HuggingFace Transformers + PEFT
- **RAG** — ChromaDB vector store, SentenceTransformers embeddings (all-MiniLM-L6-v2)
- **Data pipeline** — Python, pandas, scikit-learn
- **UI** — Streamlit
- **Containerization** — Docker
- **Environment** — Anaconda, Python 3.11

## Project Structure

```
interchange/
├── app.py                  # Streamlit application
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── data/
│   ├── raw/                # Original unmodified source data
│   ├── processed/          # Cleaned data and model results
│   └── test/               # Test inputs and ground truth answers
├── models/
│   ├── finetuned/          # LoRA adapter weights
│   └── merged/             # Fully merged model (used in production)
├── notebooks/
│   └── 01_data_exploration.ipynb
└── src/
    ├── 01_prepare_data.py      # Data cleaning and train/test split
    ├── 02_zero_shot.py         # Zero-shot baseline evaluation
    ├── 03_rag.py               # RAG pipeline evaluation
    ├── 04_finetune.py          # LoRA fine-tuning
    ├── 05_evaluate_finetuned.py # Fine-tuned model evaluation
    ├── 06_rag_finetuned.py     # RAG + fine-tuned evaluation
    └── 07_merge_model.py       # Merge LoRA weights into base model
```

## Setup and Installation

### Prerequisites

- [Anaconda](https://www.anaconda.com/download)
- [Docker Desktop](https://www.docker.com/products/docker-desktop)
- [HuggingFace account](https://huggingface.co) with access to `meta-llama/Llama-3.2-1B`

### Environment Setup

```bash
conda create -n interchange python=3.11 -y
conda activate interchange
pip install requests pandas scikit-learn openpyxl jupyter
pip install chromadb sentence-transformers langchain langchain-community
pip install transformers datasets trl torch peft
pip install streamlit
```

### Authenticate with HuggingFace

```bash
hf login
```

### Run the Pipeline

Run each script in order from `C:\projects\interchange\`:

```bash
python src/01_prepare_data.py
python src/02_zero_shot.py
python src/03_rag.py
python src/04_finetune.py
python src/05_evaluate_finetuned.py
python src/06_rag_finetuned.py
python src/07_merge_model.py
```

### Run the App Locally

```bash
streamlit run app.py
```

### Run via Docker

```bash
docker build -t interchange-app .
docker run -p 8501:8501 interchange-app
```

Open `http://localhost:8501` in your browser.

## Key Findings

Zero-shot performance confirmed this is a true interchange problem — the model cannot solve it by pattern matching alone. RAG demonstrated that retrieval significantly helps but is limited by the quality of nearest neighbors. Fine-tuning delivered the strongest results because the model directly learned the customer-to-manufacturer mapping. The combination of RAG and fine-tuning slightly regressed accuracy, likely because retrieved examples introduced conflicting signals when the fine-tuned model already had high confidence.

Remaining errors fall into four categories: part number format changes in the source data, rare edge case interchanges (e.g. automotive cross-references), suffix dropping on modifier codes, and low-frequency product categories underrepresented in training data.

## Future Improvements

- Expand training data to cover edge case categories
- Migrate pipeline to Azure Databricks for versioning and scale
- Push Docker image to Azure Container Registry
- Deploy via Azure Container Instances
- Add confidence scoring to flag low-confidence predictions for human review
