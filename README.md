# Ticket Similarity AI System

This project implements a sophisticated ticket similarity system that leverages state-of-the-art NLP models to find semantically similar support tickets. It's designed to be a powerful tool for customer support teams, helping them quickly find solutions to recurring issues.

This README provides a detailed walkthrough of the entire system, from data preprocessing to the final web application. It's intended to be an educational resource for students and anyone interested in building practical NLP applications.

## 🚀 Key Features

- **Hybrid Search:** Combines a fast bi-encoder for candidate retrieval with a more accurate cross-encoder for re-ranking, providing both speed and precision.
- **Advanced Preprocessing:** Utilizes a custom `AdaptiveMTKProcessor` that dynamically discovers patterns, keywords, and technical jargon from the data.
- **Efficient Indexing:** Employs HNSWlib for building a highly efficient approximate nearest neighbor (ANN) index, enabling real-time search even with large datasets.
- **Interactive Web App:** A Streamlit-based web application provides a user-friendly interface for searching and adding tickets.
- **Dynamic Updates:** The system can learn from new tickets, which are added to the index on the fly.

## 🧠 Core Concepts Explained

### Bi-Encoders vs. Cross-Encoders

In this project, we use two types of Transformer-based models for our search: bi-encoders and cross-encoders. Understanding their differences is key to understanding our hybrid search approach.

- **Bi-Encoders:**
  - A bi-encoder processes two pieces of text (e.g., a query and a ticket) independently. It generates a fixed-size embedding (a vector of numbers) for each piece of text.
  - To find similar items, we calculate the cosine similarity between the query embedding and the embeddings of all the tickets in our database. This is very fast, especially when combined with an ANN index like HNSW.
  - **In this project:** We use a `SentenceTransformer` model as our bi-encoder. It's used in the first stage of our search to quickly find a large set of candidate tickets.

- **Cross-Encoders:**
  - A cross-encoder, on the other hand, processes two pieces of text together as a single input. It takes both the query and a potential match and outputs a single score from 0 to 1, indicating their similarity.
  - Cross-encoders are much more accurate than bi-encoders because they can pay attention to the interactions between the two texts. However, they are also much slower, as they have to perform a full computation for every single query-ticket pair.
  - **In this project:** We use a `CrossEncoder` model in the second stage of our search. We only use it on the small set of candidates retrieved by the bi-encoder, which gives us the best of both worlds: the speed of the bi-encoder and the accuracy of the cross-encoder.

### Regular Expressions (Regex)

- A regular expression (or regex) is a sequence of characters that defines a search pattern. It's a powerful tool for finding and extracting specific patterns of text.
- **In this project:** We use regex in our `AdaptiveMTKProcessor` to discover and extract structured information from the ticket text, such as:
  - **Platform Identifiers:** A regex like `\bro\.[a-z][a-z0-9_.]{3,}\b` can find Android properties like `ro.mediatek.platform`.
  - **Error Codes:** A regex like `\b0x[0-9a-f]{3,}\b` can find hexadecimal error codes like `0xdeadbeef`.

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/):** For building the interactive web application.
- **[Sentence Transformers](https://www.sbert.net/):** For generating high-quality sentence and text embeddings.
- **[Hugging Face Transformers](https://huggingface.co/transformers/):** For accessing pre-trained language models, including the zero-shot classification model.
- **[HNSWlib](https://github.com/nmslib/hnswlib):** For building the high-performance approximate nearest neighbor search index.
- **[Pandas](https://pandas.pydata.org/):** For data manipulation and analysis.
- **[NumPy](https://numpy.org/):** For numerical operations, especially on the embeddings.
- **[Spacy](https://spacy.io/):** For natural language processing tasks like tokenization and lemmatization.
- **[Scikit-learn](https://scikit-learn.org/):** For TF-IDF vectorization.

## 🔧 How It Works

The system is divided into three main stages:

1.  **Preprocessing:** Raw ticket data is cleaned, enriched, and transformed into a format suitable for our models.
2.  **Transformation & Indexing:** The preprocessed text is converted into numerical representations (embeddings), and an efficient search index is built.
3.  **Hybrid Search & Application:** A web application allows users to search for similar tickets using a two-stage hybrid search mechanism.

### 1. Preprocessing (`preprocess.py`)

The heart of our preprocessing pipeline is the `AdaptiveMTKProcessor` class. This class is responsible for taking raw ticket data and transforming it into a feature-rich format.

**Key Steps:**

- **Pattern Discovery:** The processor first analyzes the entire dataset to discover recurring patterns using **Regular Expressions (Regex)**.
- **Keyword Extraction:** It uses TF-IDF to extract important keywords and then categorizes them using a zero-shot classification model from Hugging Face.
- **Text Enrichment:** For each ticket, the title, description, and repeat steps are processed to extract the discovered patterns and keywords.
- **`enhanced_text` Creation:** A new field, `enhanced_text`, is created for each ticket. This field is a concatenation of the processed text and special tokens representing the extracted features.

### 2. Transformation & Indexing (`transform.py`)

Once the data is preprocessed, we need to convert it into a format that our search system can understand.

**Key Steps:**

- **Embedding Generation:** We use a **bi-encoder** (`SentenceTransformer`) to convert the `enhanced_text` of each ticket into an embedding.
- **HNSW Indexing:** We use `hnswlib` to build an HNSW index for fast similarity search.
- **Artifact Storage:** The embeddings, DataFrame, HNSW index, and metadata are saved to the `models/` directory.

### 3. Hybrid Search & Application (`app.py` and `main.py`)

The final stage is the search itself, which is exposed through a Streamlit web application.

**The Two-Stage Search Process:**

1.  **Candidate Retrieval (Bi-encoder + HNSW):** The **bi-encoder** and HNSW index are used to quickly retrieve a set of candidate tickets.
2.  **Re-ranking (Cross-encoder):** The candidates are then re-ranked using a **cross-encoder** for higher accuracy.

**The Streamlit Application (`app.py`):**

- The application provides a simple interface for searching and adding tickets.

## ⚙️ How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Preprocess the data and build the index:**
    ```bash
    python preprocess.py
    python transform.py
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## ✨ Example

Let's say a user enters the query: `"camera firmware timeout on ro.mediatek.platform"`

1.  The query is encoded into an embedding by the **bi-encoder**.
2.  The HNSW index quickly finds 50 tickets that are semantically similar to the query.
3.  The **cross-encoder** then re-ranks these 50 candidates, comparing each one directly to the query.
4.  The application displays the top 5 most similar tickets, along with their similarity scores.
