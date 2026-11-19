
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Read the PDF text
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

# Step 3: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Generate embeddings for all chunks
chunk_embeddings = model.encode(chunks)

# Step 5: Enter a question
question = input("Ask a question about your PDF: ")

# Step 6: Convert question into an embedding
question_embedding = model.encode([question])

# Step 7: Calculate similarity scores
similarity_scores = cosine_similarity(
    question_embedding,
    chunk_embeddings
)[0]

# Step 8: Find the most relevant chunk
best_match_index = similarity_scores.argmax()

print("\nMost relevant chunk:")
print(chunks[best_match_index])

print("\nSimilarity score:", similarity_scores[best_match_index])
