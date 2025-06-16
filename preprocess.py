import pandas as pd
from tqdm import tqdm  
tqdm.pandas()

# Load raw ticket data
df = pd.read_csv('data/customer_support_tickets.csv')

# Optional: initial inspection
# print(df.info())
# print(df.head())
# print(df.describe())

# Keep only relevant columns
relevant_columns = [
    "Ticket ID",
    "Product Purchased",
    "Ticket Subject",
    "Ticket Description",
    "Ticket Type",
    "Resolution",
    "Ticket Priority",
    "Ticket Status"
]

df = df[relevant_columns]

# Drop rows missing essential text fields
df.dropna(subset=["Ticket Subject", "Ticket Description"], inplace=True)

# --- Improved Preprocessing ---
import spacy
import re

# Load spaCy model without unnecessary components
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Text cleaning function

def preprocess_text(text: str) -> str:
    """
    Lowercase, remove non-alphabetic, remove stopwords, and lemmatize.
    """
    text = text.lower().strip()
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return " ".join(tokens)

# Product cleaning function
def preprocess_product(prod: str) -> str:
    """
    Lowercase and remove punctuation to normalize product names.
    """
    prod = prod.lower().strip()
    # Keep alphanumeric and spaces
    prod = re.sub(r'[^a-z0-9\s]', '', prod)
    return prod

# Apply preprocessing
print("Processing Ticket Subject...")
df["Ticket Subject"] = df["Ticket Subject"].progress_apply(preprocess_text)

print("Processing Ticket Description...")
df["Ticket Description"] = df["Ticket Description"].progress_apply(preprocess_text)

print("Processing Product Purchased...")
df["clean_product"] = df["Product Purchased"].progress_apply(preprocess_product)

# Combine subject + description for embeddings
df["clean_text"] = df["Ticket Subject"] + " " + df["Ticket Description"]

# Preview cleaned data
print(df[["Ticket ID", "clean_product", "clean_text"]].head())

# Save cleaned ticket data for downstream use
df.to_csv("data/cleaned_tickets.csv", index=False)

# Also save unique products list
products = df["Product Purchased"].unique().tolist()
import pickle
with open("data/products.pkl", "wb") as f:
    pickle.dump(products, f)

print(f"Saved {len(products)} unique products to data/products.pkl")
