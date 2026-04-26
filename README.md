# SmartResume AI - AI Resume Builder & Job Matcher with RAG

An intelligent tool that analyzes your resume against a job description and generates a tailored resume using RAG (Retrieval-Augmented Generation).Built as an educational project to demonstrate practical use of **LLM integration, RAG pipelines, prompt engineering**, and modern AI application development.

### Features
- Upload your resume (PDF supported) and paste a job description
- Get a detailed match score with gap analysis
- Receive an AI-generated tailored resume optimized for the job
- Generate a personalized, professional cover letter
- Download the tailored resume as a clean `.txt` file
- Quick demo examples and clear functionality

### Tech Stack
- Gradio (UI)
- Groq (LLM)
- Sentence-Transformers + FAISS (RAG)
- PyMuPDF (PDF parsing)

### Local Setup
```bash
conda activate smartresume
python app.py