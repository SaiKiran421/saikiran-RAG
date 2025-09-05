# ===============================
# STEP 1: Install dependencies
# ===============================
!pip install pypdf sentence-transformers faiss-cpu transformers
 
# ===============================
# STEP 2: Import libraries
# ===============================
import faiss
import re
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
 
# ===============================
# STEP 3: Load multiple PDFs
# ===============================
pdf_paths = [
    "/content/2024122349_copy.pdf",                 # First uploaded PDF
    "/content/2024-wttc-introduction-to-ai_copy.pdf" # Second uploaded PDF
]
 
def extract_text_from_pdf(path):
    text = ""
    reader = PdfReader(path)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return re.sub(r"\s+", " ", text).strip()
 
documents = []
for pdf in pdf_paths:
    text = extract_text_from_pdf(pdf)
    documents.append({"source": pdf, "text": text})
    print(f"✅ Extracted {len(text)} characters from {pdf}")
 
# ===============================
# STEP 4: Chunking
# ===============================
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks
 
all_chunks = []
for doc in documents:
    chunks = chunk_text(doc["text"])
    for i, ch in enumerate(chunks):
        all_chunks.append({"text": ch, "source": doc["source"], "chunk_id": i})
 
print("✅ Total chunks created:", len(all_chunks))
 
# ===============================
# STEP 5: Embeddings + FAISS Index
# ===============================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
texts = [c["text"] for c in all_chunks]
embeddings = embedder.encode(texts, convert_to_numpy=True)
 
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
 
id_to_meta = {i: all_chunks[i] for i in range(len(all_chunks))}
print("✅ FAISS index built with", index.ntotal, "chunks")
 
# ===============================
# STEP 6: QA Model
# ===============================
qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2")
 
# ===============================
# STEP 7: Retrieval Function
# ===============================
def retrieve_paragraphs(query, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    retrieved = [id_to_meta[idx] for idx in I[0]]
    return retrieved
 
# ===============================
# STEP 8: Interactive Q&A Loop
# ===============================
while True:
    query = input("❓ Enter your question (or type 'exit'): ")
    if query.lower() == "exit":
        print("👋 Exiting Q&A system.")
        break
 
    retrieved_chunks = retrieve_paragraphs(query)
    print("\n🤖 Relevant paragraphs:\n")
    for r in retrieved_chunks:
        print(f"📖 From {r['source']} (chunk {r['chunk_id']}):\n{r['text']}\n")
 