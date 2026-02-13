import os
import glob
import time
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv

# --- 1. Load API Key ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
    exit()

# --- 2. Custom Google Embedder (WITH BATCHING) ---
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, key: str):
        self.client = genai.Client(api_key=key)
        
    def __call__(self, input: Documents) -> Embeddings:
        all_embeddings = []
        batch_size = 90  # Keep it safely below the 100 limit
        
        # Process the inputs in chunks of 90
        for i in range(0, len(input), batch_size):
            batch = input[i:i + batch_size]
            print(f"Embedding batch {i//batch_size + 1}... ({len(batch)} chunks)")
            
            response = self.client.models.embed_content(
                model='gemini-embedding-001',
                contents=batch,
            )
            all_embeddings.extend([e.values for e in response.embeddings])
            
            # Brief pause to respect API rate limits
            time.sleep(1) 
            
        return all_embeddings

# --- 3. Initialize ChromaDB ---
print("Initializing ChromaDB with Google Embeddings...")
embedder = GeminiEmbeddingFunction(key=api_key)
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="clinical_manuals",
    embedding_function=embedder
)

# --- 4. Extract and Chunk PDFs ---
pdf_folder = "./clinical_pdfs"
pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

if not pdf_files:
    print(f"No PDFs found in {pdf_folder}. Please add them and try again.")
    exit()

documents = []
ids = []
doc_id_counter = 1

print(f"Found {len(pdf_files)} PDFs. Extracting text...")

for pdf_path in pdf_files:
    print(f"Processing: {os.path.basename(pdf_path)}...")
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text = text.replace('\n', ' ').strip()
                if len(text) > 10: 
                    documents.append(text)
                    ids.append(f"chunk_{doc_id_counter}")
                    doc_id_counter += 1
    except Exception as e:
        print(f"Could not read {pdf_path}. Error: {e}")

# --- 5. Add to ChromaDB ---
if documents:
    print(f"Sending {len(documents)} total chunks to Google GenAI for embedding...")
    collection.upsert(documents=documents, ids=ids)
    print(f"Successfully embedded all {len(documents)} chunks into the database!")
else:
    print("No valid text found in PDFs.")