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
        return "❌ Missing input.", "Error", None, None
    try:
        if hasattr(resume_file, 'name') and resume_file.name.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_file)
        else:
            resume_text = extract_text_from_text(resume_file)
        
        context = build_rag_context(resume_text, job_description)
        
        # Match Analysis
        analysis_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": get_analysis_prompt(resume_text, job_description, context)}]
        )
        analysis = analysis_res.choices[0].message.content

        # Tailored Resume
        tailored_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": get_tailored_resume_prompt(resume_text, job_description)}]
        )
        tailored_resume = tailored_res.choices[0].message.content

        # Cover Letter
        cover_res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": f"Write a cover letter for:\n{resume_text[:1000]}"}]
        )
        cover_letter = cover_res.choices[0].message.content

        return analysis, tailored_resume, tailored_resume, cover_letter
    except Exception as e:
        return f"❌ {str(e)}", "Error", None, None

# ================== THE GRADIO 5 VARIABLE OVERRIDE ==================

with gr.Blocks(title="SmartResume AI", theme=gr.themes.Default()) as demo:
    
    gr.HTML("""
    <style>
        /* Force CSS Variables - Gradio 5 relies on these */
        :root, .dark, .gradio-container {
            --body-background-fill: #0a0f1c !important;
            --block-background-fill: #111827 !important;
            --input-background-fill: #1f2937 !important;
            --body-text-color: #ffffff !important;
            --heading-text-color: #67e8f9 !important;
            --block-label-text-color: #67e8f9 !important;
            --border-color-primary: #f97316 !important;
        }

        /* Fix the invisible Title and Subtitle */
        .prose h1, .prose h2, .prose h3, .prose p {
            color: #67e8f9 !important;
        }

        /* Fix the Labels (Upload Resume / Job Description) */
        .block span, label span, .block-label {
            color: #ffffff !important;
            font-weight: bold !important;
        }

        /* Fix Textbox content (currently black on dark) */
        textarea, input {
            color: #ffffff !important;
            background-color: #1f2937 !important;
        }

        /* High Contrast Primary Button */
        button.primary {
            background: #f97316 !important;
            color: white !important;
        }

        /* Force background color of the main container */
        .gradio-container { background-color: #0a0f1c !important; }
    </style>
    """)
    
    gr.Markdown("# SmartResume AI\n### AI Resume Builder & Job Matcher with RAG")

    with gr.Row():
        with gr.Column():
            resume_input = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
            job_input = gr.Textbox(label="Paste Job Description", lines=8)
        with gr.Column():
            analyze_btn = gr.Button("🚀 Analyze Resume", variant="primary")

    with gr.Row():
        match_output = gr.Textbox(label="Match Analysis", lines=10)
        tailored_output = gr.Textbox(label="Tailored Resume", lines=10)

    download_output = gr.File(label="Download Resume")
    cover_letter_output = gr.Textbox(label="📧 Cover Letter", lines=10)

    def wrap_process(resume, jd):
        analysis, tailored, _, cover = analyze_resume(resume, jd)
        if tailored and "Error" not in tailored:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(tailored)
                return analysis, tailored, f.name, cover
        return analysis, tailored, None, cover

    analyze_btn.click(wrap_process, [resume_input, job_input], [match_output, tailored_output, download_output, cover_letter_output])
    gr.Markdown("Built with Groq • Educational Portfolio")

if __name__ == "__main__":
    demo.launch()
