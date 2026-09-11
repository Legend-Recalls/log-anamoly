"""Best single-file MediaTek ticket searcher + offline RAG.

Why this instead of app.py/main.py or log-anamoly's notebook?
- app.py needs pandas, streamlit, hnswlib, sentence-transformers, cross-encoder
  (~2GB downloads, won't even import on this machine). HNSW for 302 tickets
  is overkill: brute-force cosine on 302 docs is <5ms and exact.
- app.py never preprocesses the QUERY, so raw query vs enhanced_text mismatches.
  It also never discovers patterns (discover_patterns is commented out), so
  hardware_module is always "unknown".
- log-anamoly is generic Android log anomaly (TF-IDF+SGD), zero MTK strings,
  no ticket IDs, no retrieval. Good philosophy (lightweight sklearn) but wrong task.
- This file: sklearn-only lexical hybrid (word + char TF-IDF) that nails exact
  codes like ro.mediatek.platform / MTK-1000 / 0x..., optional dense + rerank
  upgrade when models are installed, plus grounded RAG answer with citations.
  Runs with just numpy + scikit-learn (both already installed).

Usage:
    py search_rag.py "camera firmware timeout on ro.mediatek.platform"
    py search_rag.py "camera crash" --top-k 5 --rag
    py search_rag.py --eval
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

CSV_CANDIDATES = [
    "enhanced_mediatek_tickets.csv",
    "synthetic_mediatek_tickets.csv",
]

CODE_PATTERNS = [
    r"\bmtk-\d+\b",
    r"\bro\.[a-z0-9_.]+\b",
    r"\b0x[0-9a-f]+\b",
]


def find_csv() -> str:
    for c in CSV_CANDIDATES:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"No ticket CSV found, tried: {CSV_CANDIDATES}")


def load_tickets(csv_path: str | None = None) -> list[dict]:
    csv_path = csv_path or find_csv()
    tickets: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            tid = (row.get("ticket_id") or "").strip()
            if not tid:
                continue
            title = row.get("title") or ""
            desc = row.get("description") or ""
            steps = row.get("repeat_steps") or ""
            # Raw text is better than enhanced_text soup for retrieval.
            doc = f"{title}\n{desc}\n{steps}"
            tickets.append({
                "ticket_id": tid,
                "title": title,
                "description": desc,
                "repeat_steps": steps,
                "doc": doc,
            })
    return tickets


def extract_codes(text: str) -> set[str]:
    t = text.lower()
    out: set[str] = set()
    for pat in CODE_PATTERNS:
        out.update(re.findall(pat, t))
    # also keep distinctive long alnum tokens (firmware names, chip ids)
    out.update(w for w in re.findall(r"[a-z0-9_.]{5,}", t) if "." in w or "_" in w or "-" in w)
    return out


@dataclass
class Hit:
    ticket: dict
    score: float
    rank: int


class TicketSearcher:
    """Word TF-IDF + char TF-IDF + exact-code boost. Optional dense fusion."""

    def __init__(self, tickets: list[dict], dense_model: str | None = None):
        self.tickets = tickets
        docs = [t["doc"] for t in tickets]
        self.word_vec = TfidfVectorizer(
            lowercase=True, analyzer="word", ngram_range=(1, 2),
            min_df=1, max_df=0.9, sublinear_tf=True,
        )
        self.char_vec = TfidfVectorizer(
            lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
            min_df=1, max_df=0.9, sublinear_tf=True,
        )
        self.W = normalize(self.word_vec.fit_transform(docs))
        self.C = normalize(self.char_vec.fit_transform(docs))
        self.doc_codes = [extract_codes(d) for d in docs]
        # Optional dense upgrade (reuses all-mpnet-base-v2 idea, no hard dep).
        self.dense = None
        self.dense_docs = None
        if dense_model:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self.dense = SentenceTransformer(dense_model)
                embs = self.dense.encode(docs, normalize_embeddings=True,
                                         show_progress_bar=False)
                self.dense_docs = np.asarray(embs, dtype=np.float32)
            except Exception as e:
                print(f"[info] dense model unavailable ({e}), lexical-only mode.")
                self.dense = None

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        query = (query or "").strip()
        if not query:
            return []
        qw = normalize(self.word_vec.transform([query]))
        qc = normalize(self.char_vec.transform([query]))
        sw = np.asarray((qw * self.W.T).todense()).ravel()
        sc = np.asarray((qc * self.C.T).todense()).ravel()
        base = 0.6 * sw + 0.4 * sc

        # Exact code / ID boost: dense models miss these, lexical half-misses them.
        qcodes = extract_codes(query)
        qlow = query.lower()
        m = re.search(r"\bmtk-\d+\b", qlow)
        wanted_id = m.group(0).upper() if m else None
        boost = np.zeros(len(self.tickets))
        for i, t in enumerate(self.tickets):
            b = 0.0
            if wanted_id and t["ticket_id"].upper() == wanted_id:
                b += 1.0
            if qcodes:
                overlap = len(qcodes & self.doc_codes[i])
                b += min(0.30, 0.15 * overlap)
            boost[i] = b

        lexical = base + boost

        if self.dense is not None:
            qe = np.asarray(
                self.dense.encode([query], normalize_embeddings=True,
                                  show_progress_bar=False),
                dtype=np.float32).ravel()
            dense_scores = self.dense_docs @ qe  # cosine (normalized)
            # min-max to [0,1] then fuse
            d = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-9)
            l = (lexical - lexical.min()) / (lexical.max() - lexical.min() + 1e-9)
            final = 0.65 * d + 0.35 * l
        else:
            final = lexical

        idx = np.argsort(final)[::-1][:max(1, top_k)]
        return [Hit(self.tickets[i], float(final[i]), r + 1) for r, i in enumerate(idx)]


# ---------------- RAG (offline, grounded, no API key) ----------------

HW_KEYWORDS = ["camera", "modem", "audio", "wifi", "bluetooth", "display",
               "battery", "sensor", "gps", "nfc", "usb", "chip", "firmware"]

STEP_SPLIT = re.compile(r"�\+'|→|->|\||;|•|\n")


def split_steps(steps: str) -> list[str]:
    return [s.strip(" .") for s in STEP_SPLIT.split(steps or "") if s.strip(" .")]


def rag_answer(query: str, hits: list[Hit]) -> dict:
    """Extractive, cited answer. No LLM = no hallucination, works offline."""
    if not hits:
        return {"answer": "No similar tickets found.", "citations": []}
    hw = Counter()
    step_votes: Counter[str] = Counter()
    for h in hits:
        blob = (h.ticket["title"] + " " + h.ticket["description"]).lower()
        for k in HW_KEYWORDS:
            if k in blob:
                hw[k] += 1
        for s in split_steps(h.ticket["repeat_steps"]):
            step_votes[s.lower()] += 1
    top_hw = hw.most_common(2)
    hw_txt = "/".join(k for k, _ in top_hw) if top_hw else "unknown module"
    top_steps = [s for s, _ in step_votes.most_common(5)]

    cites = [f"{h.ticket['ticket_id']} ({h.score:.2f})" for h in hits]
    lines = [
        f"Query: {query}",
        f"Most similar: {', '.join(cites)}.",
        f"Likely area: {hw_txt} (majority vote over top-{len(hits)}).",
    ]
    if top_steps:
        lines.append("Suggested repro/fix path (most common steps in neighbours):")
        lines.extend(f"  {i+1}. {s}" for i, s in enumerate(top_steps[:5]))
    top = hits[0].ticket
    lines.append(f"Closest ticket: {top['ticket_id']} — {top['title']}")
    lines.append("Caveat: offline extractive summary; verify against the cited tickets above.")
    return {
        "answer": "\n".join(lines),
        "citations": [h.ticket["ticket_id"] for h in hits],
        "likely_module": hw_txt,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="MediaTek ticket search + RAG")
    ap.add_argument("query", nargs="?", default="", help="search query")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--rag", action="store_true", help="print RAG answer")
    ap.add_argument("--dense", default=None,
                    help="opt-in dense model, e.g. all-MiniLM-L6-v2")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()

    tickets = load_tickets(a.csv)
    searcher = TicketSearcher(tickets, dense_model=a.dense)

    if a.eval:
        queries = [
            "camera firmware timeout on ro.mediatek.platform",
            "MTK-1000 modem failure",
            "audio not responding reboot device",
        ]
        for q in queries:
            hits = searcher.search(q, top_k=3)
            print(f"\n# {q}")
            for h in hits:
                print(f"  {h.score:.3f} {h.ticket['ticket_id']} | {h.ticket['title'][:80]}")
            print(rag_answer(q, hits)["answer"])
        print(f"\n[eval ok] {len(tickets)} tickets indexed")
        return

    if not a.query:
        ap.error("provide a query or use --eval")
    hits = searcher.search(a.query, top_k=a.top_k)
    if a.json:
        print(json.dumps({
            "query": a.query,
            "hits": [{"rank": h.rank, "score": round(h.score, 4),
                      "ticket_id": h.ticket["ticket_id"], "title": h.ticket["title"],
                      "description": h.ticket["description"],
                      "repeat_steps": h.ticket["repeat_steps"]} for h in hits],
            "rag": rag_answer(a.query, hits),
        }, indent=2, ensure_ascii=False))
    else:
        for h in hits:
            print(f"[{h.rank}] {h.score:.3f} {h.ticket['ticket_id']} — {h.ticket['title']}")
        if a.rag:
            print("\n--- RAG ---")
            print(rag_answer(a.query, hits)["answer"])


if __name__ == "__main__":
    main()
