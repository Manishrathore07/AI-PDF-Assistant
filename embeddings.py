
from sentence_transformers import SentenceTransformer

# Step 1: Read the extracted PDF text
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

# Step 4: Convert chunks into embeddings
embeddings = model.encode(chunks)

# Step 5: Display the results
print("Total chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("First embedding:", embeddings[0])

print("\nEmbedding generation completed!")
