import fitz
import chromadb
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. READ PDF
# ==========================================

pdf_path = r"c:\Users\raush\Downloads\The_Thirsty_Crow.pdf"
document = fitz.open(pdf_path)

full_text = ""

for page in document:
    text = page.get_text()
    full_text += text + "\n"

document.close()

print("PDF read successfully!")

# ==========================================
# 2. SPLIT TEXT INTO CHUNKS
# ==========================================

chunk_size = 500

chunks = []

for i in range(0, len(full_text), chunk_size):

    chunk = full_text[i:i + chunk_size]

    if chunk.strip():
        chunks.append(chunk)

print("Total chunks:", len(chunks))

# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")

# ==========================================
# 4. CREATE EMBEDDINGS
# ==========================================

print("Creating embeddings...")

embeddings = model.encode(chunks).tolist()

print("Embeddings created!")

# ==========================================
# 5. CREATE CHROMADB
# ==========================================

print("Creating vector database...")

client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_or_create_collection(
    name="knowledge_base"
)

# ==========================================
# 6. CREATE UNIQUE IDs
# ==========================================

ids = []

for i in range(len(chunks)):
    ids.append(f"chunk_{i}")

# ==========================================
# 7. STORE DATA IN CHROMADB
# ==========================================

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)

# ==========================================
# 8. SHOW RESULT
# ==========================================

print()
print("======================================")
print("SUCCESS!")
print("======================================")

print("Total chunks stored:",
      collection.count())

print("Vector database location:")
print("data/chroma_db")

print("======================================")