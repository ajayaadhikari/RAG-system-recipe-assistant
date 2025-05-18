import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

# === Step 1: Load the CSV ===
df = pd.read_csv("13k-recipes.csv")

# Optional: Drop rows with missing data
df = df.dropna(subset=["Title", "Ingredients", "Instructions"])

# === Step 2: Prepare text for embedding ===
texts = [
    f"{row['Title']}. Ingredients: {row['Ingredients']}"
    for _, row in df.iterrows()
]

# === Step 3: Create embeddings ===
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding recipes...")
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

# === Step 4: Build FAISS index ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# === Step 5: Save index and metadata ===
os.makedirs("vector_store", exist_ok=True)

faiss.write_index(index, "vector_store/recipes.index")
df[["Title", "Ingredients", "Instructions"]].to_csv("vector_store/recipes_meta.csv", index=False)

print(f"✅ Stored {len(embeddings)} recipe embeddings to 'vector_store/'")
