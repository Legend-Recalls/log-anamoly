import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from annoy import AnnoyIndex #we will be using annoy to find nearest neighbours
import pickle

df = pd.read_csv("data/cleaned_tickets.csv")

#convert dataframe to python list
texts = df["clean_text"].tolist()

model = SentenceTransformer('sentence-t5-base')
embeddings = model.encode(texts, show_progress_bar=True)


np.save("data/ticket_embeddings.npy", embeddings) #save vector embeddings
df.to_pickle("data/tickets_df.pkl") #save dataframe

# annoy index
embedding_dim = embeddings.shape[1] # 1st field of shape gives u dimension of embedding vector
annoy_index = AnnoyIndex(embedding_dim, metric='angular') #metric measured in angular distance (since we using cosine we need angular)

for i, emb in enumerate(embeddings): #split the vector into each row
    annoy_index.add_item(i, emb) #add it to index
annoy_index.build(10) #10 trees for now
annoy_index.save("models/ticket_annoy_index.ann")
