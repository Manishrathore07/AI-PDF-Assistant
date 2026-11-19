
# Step 1: Read the extracted text
with open("extracted_text/output.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Step 2: Define chunk size and overlap
chunk_size = 500
overlap = 100

# Step 3: Create chunks
chunks = []

start = 0

while start < len(text):
    end = start + chunk_size

    chunk = text[start:end]

    chunks.append(chunk)

    start = end - overlap

# Step 4: Display the chunks
print("Total characters:", len(text))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)

# Step 5: Save chunks into a file
with open("extracted_text/chunks.txt", "w", encoding="utf-8") as file:
    for i, chunk in enumerate(chunks):
        file.write(f"\n--- Chunk {i + 1} ---\n")
        file.write(chunk + "\n")

print("\nChunking completed!")
