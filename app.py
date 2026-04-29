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
    if not resume_file or not job_description:
        return "❌ Missing input.", "", None, ""
    try:
        if hasattr(resume_file, 'name') and resume_file.name.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_file)
        else:
            resume_text = extract_text_from_text(resume_file)
        
        context = build_rag_context(resume_text, job_description)
        
        analysis_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": get_analysis_prompt(resume_text, job_description, context)}]
        )
        analysis = analysis_res.choices[0].message.content

        tailored_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": get_tailored_resume_prompt(resume_text, job_description)}]
        )
        tailored_resume = tailored_res.choices[0].message.content

        cover_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"Write a professional cover letter for this resume:\n{resume_text[:1000]}"}]
        )
        cover_letter = cover_res.choices[0].message.content

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(tailored_resume)
            tmp_path = tmp.name

        return analysis, tailored_resume, tmp_path, cover_letter
    except Exception as e:
        return f"❌ Error: {str(e)}", "", None, ""

# --- THE CLEANEST UI FIX ---
# We use 'Base' theme to stop Gradio from applying its own dark/light logic
with gr.Blocks(title="SmartResume AI", theme=gr.themes.Base()) as demo:
    
    # This style block is designed to be "un-ignorable"
    gr.HTML("""
    <style>
        /* Force a clean, professional White/Light Grey aesthetic */
        :root, .gradio-container, body {
            background-color: #f8fafc !important; 
            color: #1e293b !important;
        }

        /* Force ALL text to be dark grey/black so it is visible on white */
        .prose h1, .prose h2, .prose h3, .prose p, span, label, .markdown-text {
            color: #0f172a !important;
        }

        /* Style the input boxes so they look modern */
        textarea, input {
            background-color: white !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
        }

        /* Emerald Primary Button */
        button.primary {
            background-color: #10b981 !important;
            color: white !important;
            border: none !important;
        }
    </style>
    """)

    gr.Markdown("# 📄 SmartResume AI")
    gr.Markdown("### Professional Resume Optimizer & Job Matcher")

    with gr.Tabs():
        with gr.TabItem("1. Inputs"):
            with gr.Row():
                resume_input = gr.File(label="Upload Resume (PDF)")
                job_input = gr.Textbox(label="Job Description", lines=10)
            
            analyze_btn = gr.Button("🚀 Run Analysis", variant="primary")
            
            with gr.Row():
                sample_btn = gr.Button("Sample Data")
                clear_btn = gr.Button("Clear All", variant="stop")

        with gr.TabItem("2. Analysis"):
            match_output = gr.Textbox(label="Skill Gaps & Suggestions", lines=15)

        with gr.TabItem("3. Final Resume & Cover Letter"):
            with gr.Row():
                tailored_output = gr.Textbox(label="Optimized Resume", lines=15)
                cover_letter_output = gr.Textbox(label="Cover Letter", lines=15)
            download_output = gr.File(label="Download Resume")

    gr.Markdown("---")
    gr.Markdown("Built with Groq + RAG Pipeline")

    # Functions
    analyze_btn.click(analyze_resume, [resume_input, job_input], [match_output, tailored_output, download_output, cover_letter_output])
    sample_btn.click(lambda: (None, "Looking for a Python Developer."), outputs=[resume_input, job_input])
    clear_btn.click(lambda: (None, "", "", "", None, ""), outputs=[resume_input, job_input, match_output, tailored_output, download_output, cover_letter_output])

if __name__ == "__main__":
    demo.launch()
