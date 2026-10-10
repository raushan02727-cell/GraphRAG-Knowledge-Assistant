import chromadb
from sentence_transformers import SentenceTransformer

# --------------------------------
# 1. Load embedding model
# --------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------
# 2. Connect to ChromaDB
# --------------------------------

client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_collection(
    name="knowledge_base"
)

# --------------------------------
# 3. Ask a question
# --------------------------------

question = input("Ask a question about your PDF: ")

# --------------------------------
# 4. Convert question into embedding
# --------------------------------

question_embedding = model.encode(
    [question]
).tolist()

# --------------------------------
# 5. Search the database
# --------------------------------

results = collection.query(
    query_embeddings=question_embedding,
    n_results=3
)

# --------------------------------
# 6. Show results
# --------------------------------

print("\n==============================")
print("MOST RELEVANT INFORMATION")
print("==============================")

for i, document in enumerate(results["documents"][0]):

    print("\n--- Result", i + 1, "---")
    print(document)

print("\n==============================")