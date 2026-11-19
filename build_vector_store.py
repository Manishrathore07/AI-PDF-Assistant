
import faiss
import numpy as np
import json
import os

from sentence_transformers import SentenceTransformer

# Step 1: Read extracted PDF text
with open("extracted_text/output.txt", "r", encoding="utf-8") as file:
    text = file.read()


# Step 2: Create chunks without cutting lines

lines = text.splitlines()

chunks = []
current_chunk = ""

chunk_size = 500

for line in lines:

    line = line.strip()

    if not line:
        continue

    if len(current_chunk) + len(line) <= chunk_size:

        current_chunk += line + "\n"

    else:

        if current_chunk:
            chunks.append(current_chunk)

        current_chunk = line + "\n"

# Save the remaining text
if current_chunk:
    chunks.append(current_chunk)

# Step 3: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Generate embeddings
embeddings = model.encode(chunks)
embeddings = np.array(embeddings, dtype="float32")

# Step 5: Normalize embeddings
faiss.normalize_L2(embeddings)

# Step 6: Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

# Step 7: Add vectors
index.add(embeddings)

# Step 8: Create vector_store folder
os.makedirs("vector_store", exist_ok=True)

# Step 9: Save FAISS index
faiss.write_index(index, "vector_store/faiss_index.bin")

# Step 10: Save original chunks
with open("vector_store/chunks.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=4)

print("Vector store created successfully!")
print("Total vectors saved:", index.ntotal)
print("Total chunks saved:", len(chunks))
