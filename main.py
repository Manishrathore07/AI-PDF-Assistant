
from pypdf import PdfReader

# Step 1: Open the PDF file
reader = PdfReader("data/sample.pdf")

# Step 2: Count the pages
print("Total pages:", len(reader.pages))

# Step 3: Extract text from every page
all_text = ""

for page_number, page in enumerate(reader.pages):
    text = page.extract_text()

    if text:
        all_text += text + "\n"

# Step 4: Display extracted text
print("\nExtracted PDF Text:\n")
print(all_text)

# Step 5: Save extracted text into a file
with open("extracted_text/output.txt", "w", encoding="utf-8") as file:
    file.write(all_text)

print("\nText extraction completed!")
