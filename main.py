import numpy as np
import pandas as pd
import hnswlib
from sentence_transformers import SentenceTransformer, CrossEncoder

# —————————————————————————————
# 1) LOAD EVERYTHING (once at startup)
# —————————————————————————————

# Bi‑encoder for fast retrieval
bi_encoder = SentenceTransformer("all-mpnet-base-v2")

# Cross‑encoder for re‑ranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Ticket data + embeddings + HNSW index
df     = pd.read_pickle("models/enhanced_tickets_df.pkl")
embeds = np.load("models/enhanced_ticket_embeddings.npy")
index  = hnswlib.Index(space='cosine', dim=embeds.shape[1])
index.load_index("models/enhanced_ticket_hnsw_index.bin")
index.set_ef(50)

# —————————————————————————————
# 2) HYBRID SEARCH FUNCTION
# —————————————————————————————

def search_with_cross_encoder(
    query: str,
    top_k:      int = 5,
    hnsw_k:     int = 50
) -> pd.DataFrame:
    # Step A: retrieve fast candidates (bi‑encoder + HNSW)
    q_emb    = bi_encoder.encode([query], normalize_embeddings=True)
    labels, dists = index.knn_query(q_emb, k=hnsw_k)
    cand_idxs = labels[0]
    
    # Step B: prepare cross‑encoder inputs
    # We'll feed [query, enhanced_text] to the cross‑encoder
    cand_texts = df.iloc[cand_idxs]["enhanced_text"].tolist()
    cross_inputs = [[query, txt] for txt in cand_texts]
    
    # Step C: score with cross‑encoder
    cross_scores = cross_encoder.predict(cross_inputs)
    
    # Step D: pick top_k by cross_scores
    top_n = np.argsort(cross_scores)[::-1][:top_k]
    sel_idxs = cand_idxs[top_n]
    
    # Step E: assemble results
    results = df.iloc[sel_idxs].copy()
    results["cross_score"] = cross_scores[top_n]
    return results.reset_index(drop=True)

# —————————————————————————————
# 3) USAGE EXAMPLE
# —————————————————————————————

if __name__ == "__main__":
    q = "camera firmware timeout on ro.mediatek.platform"
    matches = search_with_cross_encoder(q, top_k=5, hnsw_k=50)
    print(matches[[
        "ticket_id","title","cross_score"
    ]])
