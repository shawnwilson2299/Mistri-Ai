from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

try:
    # Test with a simple embedding call
    response = client.embeddings.create(
        input="test",
        model="text-embedding-3-small"
    )
    print("✅ API Key is VALID!")
    print(f"✅ Successfully generated {len(response.data[0].embedding)} dimensional embedding")
except Exception as e:
    print(f"❌ API Key is INVALID or has issues:")
    print(f"Error: {str(e)}")
