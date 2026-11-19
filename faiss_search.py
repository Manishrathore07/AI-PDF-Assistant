
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Step 1: Read extracted PDF text
with open("extracted_text/output.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Step 2: Divide text into chunks
chunk_size = 500
overlap = 100

chunks = []
start = 0

while start < len(text):
    end = start + chunk_size
    chunks.append(text[start:end])
    start = end - overlap

# Step 3: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Generate embeddings
chunk_embeddings = model.encode(chunks)

# FAISS requires float32 vectors
chunk_embeddings = np.array(chunk_embeddings, dtype="float32")

# Step 5: Normalize vectors for cosine similarity
faiss.normalize_L2(chunk_embeddings)

# Step 6: Create FAISS index
dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

# Step 7: Add embeddings to the index
index.add(chunk_embeddings)

print("Total vectors stored:", index.ntotal)

# Step 8: Ask a question
question = input("\nAsk a question about your PDF: ")

# Step 9: Convert question into embedding
question_embedding = model.encode([question])

question_embedding = np.array(
    question_embedding,
    dtype="float32"
)

faiss.normalize_L2(question_embedding)

# Step 10: Search for the 3 most similar chunks
k = min(3, len(chunks))

scores, indices = index.search(question_embedding, k)

# Step 11: Display results
print("\nMost relevant chunks:\n")

for i in range(k):
    chunk_index = indices[0][i]
    score = scores[0][i]

    print(f"Result {i + 1}")
    print("Similarity score:", score)
    print(chunks[chunk_index])
    print("-" * 50)

