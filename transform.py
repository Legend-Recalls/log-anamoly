import numpy as np
import pandas as pd
import hnswlib
import pickle
import json
from sentence_transformers import SentenceTransformer

# --- Paths & Config ---
CSV_FILE = "enhanced_mediatek_tickets.csv"
EMB_FILE = "models/enhanced_ticket_embeddings.npy"
PKL_FILE = "models/enhanced_tickets_df.pkl"
INDEX_FILE = "models/enhanced_ticket_hnsw_index.bin"
ID_MAP_FILE = "models/index_id_map.pkl"
META_FILE = "models/meta_info.json"
MODEL_NAME = "all-mpnet-base-v2"
PADDING_RATIO = 0.2

# --- Load Ticket Data ---
print("📄 Loading ticket data...")
df = pd.read_csv(CSV_FILE)
texts = df["enhanced_text"].tolist()
num_elements = len(texts)

# --- Load SentenceTransformer & Generate Embeddings ---
print(f"🔗 Loading transformer model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print("🔍 Encoding tickets...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)

# --- Sanity Check ---
dim = embeddings.shape[1]
assert len(embeddings) == num_elements, "❌ Embedding count mismatch"

# --- Save Embeddings & Ticket Data ---
print("💾 Saving embeddings and DataFrame...")
np.save(EMB_FILE, embeddings)
df.to_pickle(PKL_FILE)

# --- Build HNSW Index ---
padding = int(num_elements * PADDING_RATIO)
print(f"📦 Initializing HNSW index (max_elements={num_elements + padding})...")
index = hnswlib.Index(space="cosine", dim=dim)
index.init_index(
    max_elements=num_elements + padding,
    ef_construction=200,
    M=16
)
index.add_items(embeddings, ids=np.arange(num_elements))
index.set_ef(50)
index.save_index(INDEX_FILE)

# --- Save ID Mapping ---
print("🔖 Saving index-to-ticket ID map...")
id_map = {i: row['ticket_id'] for i, row in df.iterrows()}
with open(ID_MAP_FILE, "wb") as f:
    pickle.dump(id_map, f)

# --- Save Metadata ---
print("📝 Saving model metadata...")
meta = {
    "model_name": MODEL_NAME,
    "embedding_dim": dim,
    "num_tickets": num_elements,
    "padding": padding,
    "preprocessing_version": "v2.1-enhanced-mtk"
}
with open(META_FILE, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ Index built with {num_elements} tickets (+{padding} padding)")
