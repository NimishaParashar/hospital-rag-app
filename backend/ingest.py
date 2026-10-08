import os
import sys
from dotenv import load_dotenv

# Load variables from backend/.env into environment
load_dotenv()

# SQLite fix for Linux / macOS environments
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Read Gemini API key
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is missing in .env")

# Sample data to populate DB
SAMPLE_REVIEWS = [
    {"text": "Dr. Smith in Cardiology was incredibly attentive.", "hospital": "City Care", "department": "Cardiology"},
    {"text": "The ER wait time was over 4 hours on Saturday night.", "hospital": "City Care", "department": "Emergency"},
    {"text": "Clean facilities and polite front desk staff in Pediatrics.", "hospital": "Metro Health", "department": "Pediatrics"},
    {"text": "Billing department was confusing and double-charged me.", "hospital": "City Care", "department": "Billing"}
]

def build_vector_store():
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key
    )

    docs = [
        Document(page_content=item["text"], metadata={"hospital": item["hospital"], "department": item["department"]})
        for item in SAMPLE_REVIEWS
    ]

    chroma_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_data")
    print(f"Creating Chroma DB at {chroma_dir}...")
    
    Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=chroma_dir
    )
    print("Database built and saved to disk successfully!")

if __name__ == "__main__":
    build_vector_store()