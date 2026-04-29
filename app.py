import gradio as gr
from groq import Groq
import os
import tempfile

# Groq API Key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from utils.resume_parser import extract_text_from_pdf, extract_text_from_text
from utils.rag_pipeline import build_rag_context
from utils.prompts import get_analysis_prompt, get_tailored_resume_prompt

def analyze_resume(resume_file, job_description):
    if resume_file is None or not job_description:
        return "❌ Missing input.", "No resume or job description provided.", None, "Error"
    
    try:
        if hasattr(resume_file, 'name') and resume_file.name.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_file)
        else:
            resume_text = extract_text_from_text(resume_file)
        
        context = build_rag_context(resume_text, job_description)
        
        # 1. Match Analysis
        analysis_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": get_analysis_prompt(resume_text, job_description, context)}]
        )
        analysis = analysis_res.choices[0].message.content

        # 2. Tailored Resume
        tailored_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": get_tailored_resume_prompt(resume_text, job_description)}]
        )
        tailored_resume = tailored_res.choices[0].message.content

        # 3. Cover Letter
        cover_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"Write a professional cover letter for the following resume and job description:\n\nResume: {resume_text[:1500]}\n\nJD: {job_description}"}]
        )
        cover_letter = cover_res.choices[0].message.content

        # Save tailored resume to a temp file for download
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(tailored_resume)
            tmp_path = tmp.name

        return analysis, tailored_resume, tmp_path, cover_letter
    except Exception as e:
        return f"❌ {str(e)}", "An error occurred during generation.", None, "Error"

# --- UI Setup ---
with gr.Blocks(title="SmartResume AI", theme=gr.themes.Soft(primary_hue="emerald")) as demo:
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# 📄 SmartResume AI")
            gr.Markdown("### Optimize your job search with RAG-powered AI.")
        
    with gr.Sidebar():
        gr.Markdown("## 🛠️ Inputs")
        resume_input = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
        job_input = gr.Textbox(label="Job Description", lines=10, placeholder="Paste the full job description here...")
        
        analyze_btn = gr.Button("🚀 Analyze & Generate", variant="primary")
        
        with gr.Row():
            example_btn = gr.Button("Sample Data", variant="secondary")
            clear_btn = gr.Button("Clear All", variant="stop")

    with gr.Column():
        with gr.Tabs():
            with gr.TabItem("📊 Match Analysis"):
                match_output = gr.Textbox(label="Gap Analysis & Suggestions", lines=20, show_copy_button=True)
                
            with gr.TabItem("📝 Tailored Resume"):
                tailored_output = gr.Textbox(label="AI-Optimized Resume", lines=20, show_copy_button=True)
                download_output = gr.File(label="Download Resume (.txt)")
                
            with gr.TabItem("📧 Cover Letter"):
                cover_letter_output = gr.Textbox(label="Generated Cover Letter", lines=20, show_copy_button=True)

    # --- Footer ---
    gr.Markdown("---")
    gr.Markdown("Built with **Groq + Llama 3.1** • RAG Pipeline • Educational Portfolio Project")

    # --- Functions ---
    def load_sample():
        sample_jd = "We are looking for a Senior Developer with Python, RAG, and AI integration experience."
        return None, sample_jd

    def clear_all():
        return None, "", "", "", None, ""

    analyze_btn.click(
        analyze_resume, 
        [resume_input, job_input], 
        [match_output, tailored_output, download_output, cover_letter_output]
    )
    
    example_btn.click(load_sample, outputs=[resume_input, job_input])
    clear_btn.click(clear_all, outputs=[resume_input, job_input, match_output, tailored_output, download_output, cover_letter_output])

if __name__ == "__main__":
    demo.launch()
