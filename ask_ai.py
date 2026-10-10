import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# ==========================================
# 1. Load embedding model
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded!")


# ==========================================
# 2. Connect to ChromaDB
# ==========================================

print("Connecting to vector database...")

client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_collection(
    name="knowledge_base"
)

print("Vector database connected!")
print("Documents in database:", collection.count())


# ==========================================
# 3. Ask a question
# ==========================================

question = input("\nAsk a question about your PDF: ")


# ==========================================
# 4. Convert question into embedding
# ==========================================

print("\nSearching the document...")

question_embedding = embedding_model.encode(
    [question]
).tolist()


# ==========================================
# 5. Search ChromaDB
# ==========================================

results = collection.query(
    query_embeddings=question_embedding,
    n_results=3
)

documents = results["documents"][0]


# ==========================================
# 6. Prepare context
# ==========================================

context = "\n\n---\n\n".join(documents)


# ==========================================
# 7. Create prompt
# ==========================================

prompt = f"""
You are a helpful AI research assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context,
say:

"I could not find this information in the document."

Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

Give a clear and simple answer.
"""


# ==========================================
# 8. Ask Ollama
# ==========================================

print("Generating answer using Ollama...")

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# ==========================================
# 9. Get answer
# ==========================================

answer = response["message"]["content"]


# ==========================================
# 10. Display answer
# ==========================================

print("\n======================================")
print("AI ANSWER")
print("======================================")

print(answer)


# ==========================================
# 11. Display retrieved sources
# ==========================================

print("\n======================================")
print("SOURCES USED")
print("======================================")

for i, document in enumerate(documents):

    print(f"\n--- Source {i + 1} ---")

    print(document[:500])


print("\n======================================")
print("DONE")
print("======================================")