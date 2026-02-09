import os

from dotenv import load_dotenv
from llama_parse import LlamaParse

# Load API keys from .env
load_dotenv()

# Initialize parser once
parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    result_type="markdown",
    verbose=True,
)

def parse_and_save(pdf_filename: str, output_md: str):
    """Parse one PDF and save markdown to a file."""
    print(f"🔄 Parsing {pdf_filename} ...")
    documents = parser.load_data(pdf_filename)

    with open(output_md, "w", encoding="utf-8") as f:
        for doc in documents:
            # LlamaParse already returns markdown with '---' as separators
            f.write(doc.text)
            f.write("\n---\n")

    print(f"✅ Done! Saved to '{output_md}'")

if __name__ == "__main__":
    # 1) Samsung
    parse_and_save("samsung_manual.pdf", "parsed_samsung.md")

    # 2) Whirlpool
    parse_and_save("whirlpool_manual.pdf", "parsed_whirlpool.md")

    # 3) LG
    parse_and_save("lg_manual.pdf", "parsed_lg.md")

    print("✅ All manuals parsed.")
