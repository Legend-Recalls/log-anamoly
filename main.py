import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# — Load your precomputed data & model —
df         = pd.read_pickle("data/tickets_df.pkl")
embeddings = np.load("data/ticket_embeddings.npy")
model      = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
with open("data/products.pkl","rb") as f:
    products = pickle.load(f)



# — Your existing hybrid product detector —
from transformers import pipeline
from rapidfuzz import fuzz

zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
def detect_products(query):
    nli = zero_shot(query, products)
    sem_matches = [lbl for lbl, scr in zip(nli["labels"], nli["scores"]) if scr >= 0.7]
    fuzzy = [p for p in products if fuzz.partial_ratio(query.lower(), p.lower()) >= 75]
    substr = [p for p in products if any(tok in p.lower() for tok in query.lower().split())]
    # merge unique
    seen, out = set(), []
    for lst in (sem_matches, fuzzy, substr):
        for p in lst:
            if p not in seen:
                seen.add(p); out.append(p)
    return out

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise     import cosine_similarity

def predict_top_product(text, clf, top_n=1):
    proba = clf.predict_proba([text])[0]
    labels = clf.classes_
    best_idxs = np.argsort(proba)[::-1][:top_n]
    return [labels[i] for i in best_idxs]

def find_similar_tickets(query_text, n=5,
                         tfidf_top_k=20,
                         boost=5.0,
                         prod_boost=10.0):
    # 1) Use hybrid matcher to detect top-1 product
    top_matches = detect_products(query_text)
    top_product = top_matches[0] if top_matches else None
    if not top_product:
        print("❌ Could not detect a related product.")
        return pd.DataFrame()

    # 2) Filter to top product only
    mask = df["Product Purchased"] == top_product
    sub_df   = df[mask].reset_index(drop=True)
    sub_embs = embeddings[mask.values]
    if len(sub_df) == 0:
        print("⚠️ No tickets for detected product, using fallback to all.")
        sub_df, sub_embs = df.reset_index(drop=True), embeddings

    # 3) TF-IDF filtering
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(sub_df["clean_text"])
    q_tfidf = tfidf.transform([query_text])
    tfidf_sims = cosine_similarity(q_tfidf, X_tfidf)[0] * 100

    # 4) Select top-k for semantic scoring
    top_idxs = tfidf_sims.argsort()[::-1][:tfidf_top_k]
    sub_df   = sub_df.iloc[top_idxs].reset_index(drop=True)
    sub_embs = sub_embs[top_idxs]

    # 5) Semantic similarity
    q_emb     = model.encode([query_text])[0]
    sem_sims  = cosine_similarity([q_emb], sub_embs)[0] * 100

    # 6) Weighted scoring
    weight_tfidf = 0.8
    weight_sem   = 0.2
    combined = (
        weight_tfidf * tfidf_sims[top_idxs]
      + weight_sem   * sem_sims
      + (sub_df["Product Purchased"] == top_product).astype(float) * prod_boost
      + sub_df["Resolution"].notna().astype(float) * boost
    )

    sub_df["tfidf_score"]    = tfidf_sims[top_idxs]
    sub_df["semantic_score"] = sem_sims
    sub_df["final_score"]    = combined

    return sub_df.sort_values("final_score", ascending=False).head(n)[[
        "Ticket ID","Product Purchased","Ticket Subject",
        "tfidf_score","semantic_score","final_score"
    ]]

# — CLI demo —
if __name__=="__main__":
    q = input("Enter your issue: ")
    k = int(input("How many results? "))
    df_out = find_similar_tickets(q, n=k)
    print(df_out.to_string(index=False))
