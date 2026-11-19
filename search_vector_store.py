
import faiss
import json
import numpy as np

from sentence_transformers import SentenceTransformer

# Step 1: Load the saved FAISS index
index = faiss.read_index("vector_store/faiss_index.bin")

# Step 2: Load the saved chunks
with open("vector_store/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Step 3: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Vector store loaded successfully!")
print("Total vectors:", index.ntotal)
print("Total chunks:", len(chunks))

# Step 4: Ask a question
question = input("\nAsk a question about your PDF: ")

# Step 5: Convert question into embedding
question_embedding = model.encode([question])

question_embedding = np.array(
    question_embedding,
    dtype="float32"
)

# Step 6: Normalize question embedding
faiss.normalize_L2(question_embedding)

# Step 7: Search FAISS index
k = min(3, index.ntotal)

scores, indices = index.search(question_embedding, k)

# Step 8: Display results
print("\nMost relevant chunks:\n")

for i in range(k):

    chunk_index = indices[0][i]
    similarity_score = scores[0][i]

    print(f"Result {i + 1}")
    print("Similarity score:", similarity_score)
    print("\nChunk content:")
    print(chunks[chunk_index])

    print("-" * 50)
