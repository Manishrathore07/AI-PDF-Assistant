
from dotenv import load_dotenv
from google import genai
import os

# Step 1: Load environment variables
load_dotenv()

# Step 2: Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Step 3: Check whether API key exists
if not api_key:
    print("API key not found!")
else:
    print("API key loaded successfully!")

    # Step 4: Connect to Gemini
    client = genai.Client(api_key=api_key)

    # Step 5: Send a test prompt
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Explain RAG in two simple sentences."
    )

    # Step 6: Display Gemini response
    print("\nGemini Response:")
    print(response.text)
