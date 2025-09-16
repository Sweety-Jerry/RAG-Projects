pip install langchain-community

 pip install faiss-cpu

pip install chromadb

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.docstore.document import Document
import os

# ✅ Step 1: Sample texts
texts = [
    "LangChain makes building LLM apps easier.",
    "FAISS enables fast similarity search over embeddings.",
    "ChromaDB allows storing documents with metadata."
]

# ✅ Step 2: Convert texts into LangChain Document format
documents = [Document(page_content=txt) for txt in texts]

# ✅ Step 3: Load Embedding model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Step 4A: Store in FAISS
faiss_db = FAISS.from_documents(documents, embedding)
print("✅ FAISS: Stored", len(faiss_db.index.reconstruct_n(0, len(texts))), "vectors")

# ✅ Step 4B: Store in ChromaDB
chroma_db = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    collection_name="my_collection",
    persist_directory="./chroma_store"  # Optional: saves to disk
)

print("✅ ChromaDB: Stored", len(chroma_db.get()['ids']), "vectors")

# ✅ Optional: Persist ChromaDB for reuse
chroma_db.persist()
