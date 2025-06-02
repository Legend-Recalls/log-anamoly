A short, clear implementation of searching through sentence transformers and ANNOY

# Ticket Similarity AI System

This project uses Sentence Transformers and Annoy to identify semantically similar support tickets.

DATA EXTRACTED FROM [kaggle](https://https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset?resource=download)

## 🔧 How It Works

- preprocessing
- generate embedding vectors
- store embeddings in an Annoy index (tree size of 10)
- accept user queries and return similar tickets based on cosine similarity

### 1. Retrieve relevant data

### 2. Preprocess and build index

```bash
python preprocess.py
python transform.py
```

### 3. Search

```bash
python /search.py
```

## ✨ Example
