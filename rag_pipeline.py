
import faiss
import json
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai


# Step 1: Load environment variables
load_dotenv()

# Step 2: Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

# Step 3: Connect to Gemini
client = genai.Client(api_key=api_key)

# Step 4: Load FAISS index
index = faiss.read_index("vector_store/faiss_index.bin")

# Step 5: Load saved PDF chunks
with open("vector_store/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Step 6: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("RAG pipeline components loaded successfully!")


# Step 7: Ask a question
question = input("\nAsk a question about your PDF: ")

# Step 8: Convert question into embedding
question_embedding = model.encode([question])

# Step 9: Convert embedding into float32 format
question_embedding = np.array(
    question_embedding,
    dtype="float32"
)

# Step 10: Normalize the question embedding
faiss.normalize_L2(question_embedding)

# Step 11: Search for the 3 most relevant chunks
k = min(8, index.ntotal)

scores, indices = index.search(question_embedding, k)


# Step 12: Collect retrieved chunks with source IDs

retrieved_chunks = []
source_ids = []

for i in range(k):
    chunk_index = indices[0][i]

    retrieved_chunks.append(chunks[chunk_index])

    source_ids.append(chunk_index + 1)


# Step 13: Combine retrieved chunks
context = "\n\n".join(retrieved_chunks)

print("\nRelevant information retrieved successfully!")
print("\nRetrieved Context:\n")
print(context)


# Step 14: Create prompt for Gemini


prompt = f"""
You are an AI assistant that answers questions about a PDF.

Instructions:
1. Use ONLY the information provided in the context.
2. Include all relevant information related to the user's question.
3. Do not skip entries marked as Holiday or Sectional Holiday.
4. Clearly distinguish regular holidays from sectional holidays.
5. Do not invent dates or information.
6. If the answer is not available in the context, say:
   "I could not find this information in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""


# Step 15: Send context and question to Gemini

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt,
    config={
        "max_output_tokens": 300
    }
)

# Step 16: Display final answer

print("\nAI Answer:\n")
print(response.text)

# Step 17: Display source references

print("\nSources:\n")

for source_id in source_ids:
    print(f"PDF - Chunk {source_id}")
    print(chunks[source_id - 1])
    print("-" * 50)


