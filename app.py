import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from rapidfuzz import fuzz
import pickle

#use cacheing so every search doesnt take too muhc time
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

@st.cache_resource
def load_zero_shot():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def load_data():
    df = pd.read_pickle("data/tickets_df.pkl")
    emb = np.load("data/ticket_embeddings.npy")
    with open("data/products.pkl", "rb") as f:
        products = pickle.load(f)
    return df, emb, products

# — PRODUCT DETECTOR —
def detect_products(query, products, zero_shot):
    nli = zero_shot(query, products)
    sem_matches = [lbl for lbl, scr in zip(nli["labels"], nli["scores"]) if scr >= 0.7]
    fuzzy = [p for p in products if fuzz.partial_ratio(query.lower(), p.lower()) >= 75]
    substr = [p for p in products if any(tok in p.lower() for tok in query.lower().split())]

    seen, out = set(), []
    for lst in (sem_matches, fuzzy, substr):
        for p in lst:
            if p not in seen:
                seen.add(p); out.append(p)
    return out

# similarity finder
def find_similar_tickets(query_text, df, embeddings, products, model, zero_shot,
                         n=5, tfidf_top_k=20, boost=5.0, prod_boost=10.0):
    top_matches = detect_products(query_text, products, zero_shot)
    top_product = top_matches[0] if top_matches else None
    if not top_product:
        st.warning("❌ Could not detect a related product.")
        return pd.DataFrame()

    mask = df["Product Purchased"] == top_product
    sub_df   = df[mask].reset_index(drop=True)
    sub_embs = embeddings[mask.values]
    if len(sub_df) == 0:
        st.warning("⚠️ No tickets for detected product. Falling back to all.")
        sub_df, sub_embs = df.reset_index(drop=True), embeddings

    # TF-IDF
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(sub_df["clean_text"])
    q_tfidf = tfidf.transform([query_text])
    tfidf_sims = cosine_similarity(q_tfidf, X_tfidf)[0] * 100

    # Top-K
    top_idxs = tfidf_sims.argsort()[::-1][:tfidf_top_k]
    sub_df   = sub_df.iloc[top_idxs].reset_index(drop=True)
    sub_embs = sub_embs[top_idxs]

    # S
    q_emb     = model.encode([query_text])[0]
    sem_sims  = cosine_similarity([q_emb], sub_embs)[0] * 100

    # Combine scores
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

st.title(" Smart Ticket Finder")

query = st.text_input("Enter your issue/query:", placeholder="e.g., xbox controller firmware not updating")
num_results = st.slider("Number of results", 1, 20, 5)

if query:
    with st.spinner("Searching..."):
        df, embeddings, products = load_data()
        model      = load_model()
        zero_shot  = load_zero_shot()

        result_df = find_similar_tickets(
            query_text=query,
            df=df,
            embeddings=embeddings,
            products=products,
            model=model,
            zero_shot=zero_shot,
            n=num_results
        )
        if not result_df.empty:
            st.success("Top matching tickets:")
            st.dataframe(result_df)
    

    st.subheader("📄 View Full Ticket Details")
    sel = st.selectbox("Pick a Ticket ID", result_df["Ticket ID"].tolist())
    if sel:
        t = df[df["Ticket ID"]==sel].iloc[0]
        st.markdown(f"""
**🎫 ID:** {t['Ticket ID']}  
**💻 Product:** {t['Product Purchased']}  
**📌 Subject:** {t['Ticket Subject']}  
**📝 Description:** {t['Ticket Description']}  
**✅ Resolution:** {t['Resolution'] or '—'}  
""")
