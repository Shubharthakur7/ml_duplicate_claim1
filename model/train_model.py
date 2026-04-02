import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

data = pd.read_csv("../data/claims.csv")

data["text"] = data["title"] + " " + data["description"]

texts = data["text"].tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts)

np.save("claims_embeddings.npy", embeddings)

print("Claims embeddings created successfully")
print(f"Total claims encoded: {len(texts)}")
