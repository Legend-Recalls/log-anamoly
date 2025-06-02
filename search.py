import numpy as np
import pandas as pd
from annoy import AnnoyIndex 
import pickle
from sentence_transformers import SentenceTransformer

#load existing data
df = pd.read_pickle("data/tickets_df.pkl")
embeddings = np.load("data/ticket_embeddings.npy")
annoy_index = AnnoyIndex(embeddings.shape[1], metric='angular')
annoy_index.load("models/ticket_annoy_index.ann")
model = SentenceTransformer('sentence-t5-base')

def find_similar_tickets(query_text, n=5):
    query_embedding = model.encode([query_text])[0]
    indices, distances = annoy_index.get_nns_by_vector(query_embedding, n, include_distances=True)
    
    results = df.iloc[indices].copy() #select row by integer
    similarities = [1 - (d ** 2) / 2 for d in distances] #cosine similarity formula
    results["similarity_score"] = similarities
    
    return results[["Ticket ID", "Ticket Subject", "Resolution", "Ticket Priority", "similarity_score"]]


query = input("Enter your issue: ")
numberfetches = int(input("Enter no. of maximum fetches to do: "))

results = find_similar_tickets(query, numberfetches)
print(results)
