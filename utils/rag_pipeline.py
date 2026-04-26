from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def build_rag_context(resume_text, job_description):
    """Build RAG context by retrieving relevant resume chunks"""
    # Split resume into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(resume_text)
    
    if not chunks:
        return ""
    
    # Create embeddings
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    
    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    # Search for relevant chunks
    query_embedding = embedder.encode([job_description])
    D, I = index.search(query_embedding.astype('float32'), k=min(5, len(chunks)))
    
    relevant_chunks = [chunks[i] for i in I[0]]
    context = "\n\n".join(relevant_chunks)
    
    return context