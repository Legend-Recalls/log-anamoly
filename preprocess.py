import pandas as pd
from tqdm import tqdm  
tqdm.pandas()

df = pd.read_csv('data/customer_support_tickets.csv')

# print(df.info())
# print(df.head())
# print(df.describe())


#removed stuff that was going to be unique and non redundant most of the time
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


#filter columns to contain only relevant columns and remove different ones
df = df[relevant_columns]

#filter nan columns
df.dropna(subset=["Ticket Subject", "Ticket Description"], inplace=True) #from testing i found out that these fields were there in almost every ticket so this is not needed

# print(df.info())



import spacy

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess(text):
    doc = nlp(text.lower())  # lowercase
    tokens = [
        token.lemma_                 # lemmatization
        for token in doc
        if token.is_alpha           # remove punctuation/numbers
        and not token.is_stop       # remove stopwords
    ]
    return " ".join(tokens)

print("Processing Ticket Subject...")
df["Ticket Subject"] = df["Ticket Subject"].progress_apply(preprocess)
print("Processing Ticket Description...")
df["Ticket Description"] = df["Ticket Description"].progress_apply(preprocess)


df["clean_text"] = df["Ticket Subject"] + " " + df["Ticket Description"]


print(df[["Ticket ID", "Ticket Subject", "Ticket Description", "clean_text"]].head())
df.to_csv("data/cleaned_tickets.csv", index=False)
