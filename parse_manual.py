import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

# Load API keys
load_dotenv()

# Initialize parser
parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="markdown",
    verbose=True
)

# Parse the Samsung manual
print("🔄 Parsing Samsung manual...")
documents = parser.load_data("samsung_manual.pdf")

# Save output
with open("parsed_manual.md", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.text)
        f.write("\n---\n")

print("✅ Done! Check 'parsed_manual.md'")
