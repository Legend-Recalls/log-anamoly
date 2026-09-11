import os
import uuid
import json
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer, CrossEncoder

# Import your processor
from preprocess import AdaptiveMTKProcessor

CSV_FILE = "enhanced_mediatek_tickets.csv"
PKL_FILE = "models/enhanced_tickets_df.pkl"
EMB_FILE = "models/enhanced_ticket_embeddings.npy"
INDEX_FILE = "models/enhanced_ticket_hnsw_index.bin"
ID_MAP_FILE = "models/index_id_map.pkl"
META_FILE = "models/meta_info.json"

# Instantiate a single processor
processor = AdaptiveMTKProcessor()

@st.cache_resource
def load_bi_encoder():
    return SentenceTransformer("all-mpnet-base-v2")

@st.cache_resource
def load_cross_encoder():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def load_data_and_index():
    df = pd.read_pickle(PKL_FILE)
    embeddings = np.load(EMB_FILE)
    index = hnswlib.Index(space='cosine', dim=embeddings.shape[1])
    index.load_index(INDEX_FILE)
    index.set_ef(200)
    return df, embeddings, index

def search_with_cross_encoder(query, bi_encoder, cross_encoder, df, index, hnsw_k=50, top_k=5):
    candidate_count = min(hnsw_k, index.get_current_count())
    if candidate_count == 0:
        return pd.DataFrame()
    index.set_ef(max(50, candidate_count))
    q_emb = bi_encoder.encode([query], normalize_embeddings=True)
    labels, _ = index.knn_query(q_emb, k=candidate_count)
    candidate_idxs = labels[0]
    candidate_texts = df.iloc[candidate_idxs]["enhanced_text"].tolist()
    cross_inputs = [[query, txt] for txt in candidate_texts]
    cross_scores = cross_encoder.predict(cross_inputs)
    top_n = np.argsort(cross_scores)[::-1][:top_k]
    selected_idxs = candidate_idxs[top_n]
    results = df.iloc[selected_idxs].copy()
    results["CrossScore"] = cross_scores[top_n]
    return results.reset_index(drop=True)

# Streamlit layout
st.set_page_config(page_title="MediaTek Hybrid Ticket Search", layout="wide")
st.title("🎯 MediaTek Hybrid Ticket Similarity Search")

# Load data & models
df, embeddings, hnsw_index = load_data_and_index()
bi_encoder = load_bi_encoder()
cross_encoder = load_cross_encoder()

# --- Search Section ---
st.header("🔍 Search Existing Tickets")
query = st.text_input("Enter your ticket query")
hnsw_k = st.slider("HNSW candidate pool size", 10, 200, 50)
top_k = st.slider("Top K results", 1, 10, 5)

if st.button("Search") and query:
    with st.spinner("Searching..."):
        results = search_with_cross_encoder(
            query, bi_encoder, cross_encoder, df, hnsw_index, hnsw_k, top_k
        )
    if results.empty:
        st.warning("No similar tickets found.")
    else:
        for i, row in results.iterrows():
            st.markdown(f"### Match #{i+1} (Score: {row['CrossScore']:.4f})")
            st.markdown(f"**🎫 ID:** {row['ticket_id']}")
            st.markdown(f"**📌 Title:** {row['title']}")
            st.markdown(f"**📝 Description:** {row['description']}")
            st.markdown(f"**🔁 Steps:** {row['repeat_steps']}")
            st.markdown("---")

# --- Add Ticket Section ---
st.header("➕ Add a New Ticket")

with st.form("add_ticket_form"):
    new_title = st.text_area(
        "Ticket Title",
        placeholder="e.g., mtk: camera crash during init"
    )
    new_description = st.text_area("Ticket Description")
    new_steps = st.text_area(
        "Repeat Steps (→ separated)",
        placeholder="e.g., reboot → flash → observe logcat"
    )

    submitted = st.form_submit_button("Add Ticket")
    if submitted and new_title and new_description and new_steps:
        with st.spinner("Processing new ticket..."):
            # Preprocess with your class instance
            title_proc = processor.preprocess_title_enhanced(new_title)
            steps_proc = processor.preprocess_steps_enhanced(new_steps)
            desc_proc  = processor.preprocess_description_enhanced([new_description])[0]

            row = {
                "ticket_id": str(uuid.uuid4())[:8],
                "title": new_title,
                "description": new_description,
                "repeat_steps": new_steps,
                "title_processed": title_proc,
                "steps_processed": steps_proc,
                "desc_processed": desc_proc
            }
            # Use the class method as well
            row["enhanced_text"] = processor.create_enhanced_features(row)

            # Embed
            new_emb = bi_encoder.encode(
                [row["enhanced_text"]], normalize_embeddings=True
            )[0]

            # Add to HNSW index
            new_idx = len(df)
            current_count = hnsw_index.get_current_count()
            if current_count >= hnsw_index.get_max_elements():
                hnsw_index.resize_index(max(current_count + 1, int(current_count * 1.5)))
            hnsw_index.add_items([new_emb], [new_idx])

            # Append to DataFrame
            df.loc[new_idx] = row

            # Save updates
            # Reload from disk so a Streamlit rerun cannot overwrite vectors
            # added by an earlier request.
            saved_embeddings = np.load(EMB_FILE)
            np.save(EMB_FILE, np.vstack([saved_embeddings, new_emb]))
            df.to_pickle(PKL_FILE)
            hnsw_index.save_index(INDEX_FILE)
            with open(ID_MAP_FILE, "rb") as f:
                id_map = pickle.load(f)
            id_map[new_idx] = row["ticket_id"]
            with open(ID_MAP_FILE, "wb") as f:
                pickle.dump(id_map, f)
            if os.path.exists(META_FILE):
                with open(META_FILE, encoding="utf-8") as f:
                    meta = json.load(f)
                meta["num_tickets"] = hnsw_index.get_current_count()
                meta["max_elements"] = hnsw_index.get_max_elements()
                meta["spare_capacity"] = meta["max_elements"] - meta["num_tickets"]
                with open(META_FILE, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            pd.DataFrame([row])[[
                "ticket_id", "title", "description", "repeat_steps", "enhanced_text"
            ]].to_csv(
                CSV_FILE,
                mode='a',
                index=False,
                header=not os.path.exists(CSV_FILE)
            )

            st.success("✅ New ticket added and saved successfully!")
            load_data_and_index.clear()
