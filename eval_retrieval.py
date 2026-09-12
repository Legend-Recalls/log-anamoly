"""Retrieval eval: 24 everyday + technical queries with known-good tickets.

Measures Recall@1/3/5, MRR and latency of the exact pipeline app.py runs
(clean query -> bi-encoder -> candidates -> cross-encoder rerank), except
candidates come from exact brute-force cosine instead of HNSW (same vectors,
so scores are an upper bound on what the approximate index returns).

Run:  py eval_retrieval.py [--top-k 5] [--hnsw-k 50] [--plot docs/eval.png]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

# query -> tickets that genuinely solve it (checked against the corpus)
QUERIES = {
    "camera firmware timeout on ro.mediatek.platform": ["MTK-2002", "MTK-2040"],
    "S_FT_DOWNLOAD_FAIL 4008": ["MTK-3170", "MTK-3172"],
    "widevine L1 dropped to L3": ["MTK-3114", "MTK-3110"],
    "bluetooth showing error": ["MTK-3126", "MTK-3200"],
    "avc denied camera hal": ["MTK-3120"],
    "kernel panic phone reboots itself": ["MTK-3130", "MTK-3131"],
    "battery stuck at 50 percent for hours": ["MTK-3380"],
    "5G icon shows but speed is 4G": ["MTK-3310", "MTK-3311"],
    "eSIM download stalls at 80 percent": ["MTK-3150", "MTK-3152"],
    "under display fingerprint enroll always fails": ["MTK-3350", "MTK-3352"],
    "UFS read errors on long video record": ["MTK-3100"],
    "watchdog timeout freeze then reboot": ["MTK-3140", "MTK-3141"],
    "wifi switch got turned off no internet": ["MTK-3400", "MTK-3401"],
    "alarm rang at night instead of morning": ["MTK-3454", "MTK-3455"],
    "screen turned black and white overnight": ["MTK-3451", "MTK-3452"],
    "phone narrates every tap out loud": ["MTK-3487", "MTK-3488"],
    "forgot PIN after months of fingerprint": ["MTK-3616", "MTK-3617"],
    "earbuds play audio from the wrong phone": ["MTK-3433", "MTK-3435"],
    "deleted holiday photos how to recover": ["MTK-3535", "MTK-3536"],
    "music stops when the screen turns off": ["MTK-3565", "MTK-3567"],
    "messages arrive hours late": ["MTK-3568", "MTK-3569"],
    "flashlight turns on by itself in pocket": ["MTK-3518", "MTK-3517"],
    "overnight charge only reached 60 percent": ["MTK-3694", "MTK-3695"],
    "alarm shows but makes no sound": ["MTK-3457", "MTK-3458"],
}


def load_index():
    try:
        import hnswlib
        have = True
    except ImportError:
        have = False
    df = pd.read_pickle("models/enhanced_tickets_df.pkl")
    embeds = np.load("models/enhanced_ticket_embeddings.npy")
    index = None
    if have:
        index = hnswlib.Index(space="cosine", dim=embeds.shape[1])
        index.load_index("models/enhanced_ticket_hnsw_index.bin")
        index.set_ef(200)
    return df, embeds, index, have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--hnsw-k", type=int, default=50)
    ap.add_argument("--plot", default="docs/eval.png")
    a = ap.parse_args()

    from preprocess import AdaptiveMTKProcessor
    from sentence_transformers import SentenceTransformer, CrossEncoder

    print("Loading models (takes a bit on CPU)...")
    processor = AdaptiveMTKProcessor()
    bi = SentenceTransformer("all-mpnet-base-v2")
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    df, embeds, index, have_hnsw = load_index()
    print(f"Tickets: {len(df)}, HNSW: {'yes' if have_hnsw else 'no (exact brute force)'}")

    rows, lat = [], []
    for q, want in QUERIES.items():
        t0 = time.time()
        tp = processor.preprocess_title_enhanced(q)
        sp = processor.preprocess_steps_enhanced(q)
        dp = processor.preprocess_description_enhanced([q])[0]
        clean = processor.create_enhanced_features(
            {"title_processed": tp, "steps_processed": sp, "desc_processed": dp})
        qe = bi.encode([clean], normalize_embeddings=True)[0]
        if have_hnsw:
            labels, _ = index.knn_query(qe, k=min(a.hnsw_k, index.get_current_count()))
            cand = labels[0]
        else:
            cand = np.argsort(embeds @ qe)[::-1][:a.hnsw_k]
        texts = df.iloc[cand]["enhanced_text"].tolist()
        scores = ce.predict([[q, t] for t in texts])
        ranked = [df.iloc[cand[i]]["ticket_id"] for i in np.argsort(scores)[::-1][:a.top_k]]
        lat.append((time.time() - t0) * 1000)

        hits = [i + 1 for i, tid in enumerate(ranked) if tid in want]
        first = min(hits) if hits else None
        rows.append({"query": q, "want": want, "got": ranked,
                     "recall@1": any(t in ranked[:1] for t in want),
                     "recall@3": any(t in ranked[:3] for t in want),
                     "recall@5": any(t in ranked[:5] for t in want),
                     "rr": 1 / first if first else 0.0})

    r1 = sum(r["recall@1"] for r in rows) / len(rows)
    r3 = sum(r["recall@3"] for r in rows) / len(rows)
    r5 = sum(r["recall@5"] for r in rows) / len(rows)
    mrr = sum(r["rr"] for r in rows) / len(rows)
    lat_sorted = sorted(lat)
    print(f"\nQueries: {len(rows)}  R@1={r1:.3f}  R@3={r3:.3f}  R@5={r5:.3f}  "
          f"MRR={mrr:.3f}  latency p50={lat_sorted[len(lat_sorted)//2]:.0f}ms")
    for r in rows:
        mark = "ok " if r["recall@5"] else "MISS"
        print(f"[{mark}] {r['query'][:52]:52s} -> {r['got'][0]}")

    os.makedirs("docs", exist_ok=True)
    with open("docs/eval_results.json", "w") as f:
        json.dump({"recall@1": r1, "recall@3": r3, "recall@5": r5, "mrr": mrr,
                   "latency_p50_ms": lat_sorted[len(lat_sorted) // 2],
                   "hnsw": have_hnsw, "rows": rows}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
    labels = [r["query"][:34] for r in rows]
    colors = ["#2ca02c" if r["recall@5"] else "#d62728" for r in rows]
    ax1.barh(labels, [r["rr"] for r in rows], color=colors)
    ax1.set_xlabel("reciprocal rank (1.0 = right ticket first)")
    ax1.set_title(f"Per-query score — R@5 {r5:.0%}, MRR {mrr:.2f}")
    ax2.bar(["R@1", "R@3", "R@5"], [r1, r3, r5], color="#1f77b4")
    ax2.set_ylim(0, 1)
    ax2.set_title("Overall recall")
    fig.tight_layout()
    fig.savefig(a.plot, dpi=110)
    print(f"Saved {a.plot} + docs/eval_results.json")


if __name__ == "__main__":
    main()
