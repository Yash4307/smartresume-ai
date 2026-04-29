import gradio as gr
from groq import Groq
import os
import tempfile

# Groq API Key from HF Spaces Secrets
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from utils.resume_parser import extract_text_from_pdf, extract_text_from_text
from utils.rag_pipeline import build_rag_context
from utils.prompts import get_analysis_prompt, get_tailored_resume_prompt

def analyze_resume(resume_file, job_description):
    if resume_file is None:
        return "❌ Please upload your resume.", "No resume provided.", None, None
    if not job_description or len(job_description.strip()) < 20:
        return "❌ Please provide a job description.", "No job description provided.", None, None
    try:
        if hasattr(resume_file, 'name') and resume_file.name.lower().endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_file)
        else:
            resume_text = extract_text_from_text(resume_file)
        
        context = build_rag_context(resume_text, job_description)
        
        analysis_prompt = get_analysis_prompt(resume_text, job_description, context)
        analysis_res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": analysis_prompt}])
        analysis = analysis_res.choices[0].message.content

        tailored_prompt = get_tailored_resume_prompt(resume_text, job_description)
        tailored_res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": tailored_prompt}])
        tailored_resume = tailored_res.choices[0].message.content

        cover_prompt = f"Write a cover letter:\nResume: {resume_text[:1000]}\nJD: {job_description}"
        cover_res = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": cover_prompt}])
        cover_letter = cover_res.choices[0].message.content

        return analysis, tailored_resume, tailored_resume, cover_letter
    except Exception as e:
        return f"❌ Error: {str(e)}", "Error", None, None

# ================== CSS FORCE FIX ==================

with gr.Blocks(
    title="SmartResume AI",
    theme=gr.themes.Default() 
) as demo:
    
    gr.HTML("""
    <style>
        /* 1. Force the page background to be dark */
        body, .gradio-container {
            background-color: #0a0f1c !important;
        }

        /* 2. Force Title and Subtitle to be Bright Cyan */
        h1, h2, h3, .prose h1, .prose h2, .prose h3 {
            color: #67e8f9 !important;
            text-align: center;
        }

        /* 3. Force Labels (Upload Resume, Job Description) to be White */
        label span, .block-label, .sr-only + span {
            color: #ffffff !important;
            font-weight: bold !important;
            font-size: 1.1rem !important;
        }

        /* 4. Force Input Text visibility */
        textarea, input {
            color: #ffffff !important;
            background-color: #1f2937 !important;
            border: 1px solid #374151 !important;
        }

        /* 5. Force Footer and small text to be Cyan */
        p, .markdown-text, .footer-text {
            color: #67e8f9 !important;
        }

        /* 6. Fix the "Built with Gradio" area */
        footer {
            display: none !important;
        }

        /* 7. Button Styling */
        button.primary {
            background: linear-gradient(90deg, #f97316, #ea580c) !important;
            color: white !important;
            border: none !important;
        }
    </style>
    """)
    
    gr.Markdown("# SmartResume AI")
    gr.Markdown("### AI Resume Builder & Job Matcher with RAG")

    with gr.Row():
        with gr.Column(scale=1):
            resume_input = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
            job_input = gr.Textbox(label="Paste Job Description", lines=8, placeholder="Paste JD here...")

        with gr.Column(scale=1):
            analyze_btn = gr.Button("🚀 Analyze & Generate Tailored Resume", variant="primary", size="large")

    with gr.Row():
        example_btn = gr.Button("Load Sample Data", variant="secondary")
        clear_btn = gr.Button("Clear All", variant="stop")

    with gr.Row():
        with gr.Column(scale=1):
            match_output = gr.Textbox(label="Match Analysis & Gap Suggestions", lines=13)
        with gr.Column(scale=1):
            tailored_output = gr.Textbox(label="Tailored Resume", lines=13)

    download_output = gr.File(label="Download Tailored Resume (.txt)", visible=True)
    cover_letter_output = gr.Textbox(label="📧 Generated Cover Letter", lines=12)

    def process_resume(resume_file, job_description):
        analysis, tailored_text, _, cover_letter = analyze_resume(resume_file, job_description)
        if tailored_text and "Error" not in tailored_text:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                tmp.write(tailored_text)
                return analysis, tailored_text, tmp.name, cover_letter
        return analysis, tailored_text, None, cover_letter

    analyze_btn.click(process_resume, [resume_input, job_input], [match_output, tailored_output, download_output, cover_letter_output])
    example_btn.click(lambda: (None, "Sample JD: Senior Python Developer."), None, [resume_input, job_input])
    clear_btn.click(lambda: (None, "", "", None, ""), None, [resume_input, job_input, match_output, download_output, cover_letter_output])

    gr.Markdown("Built with Groq + RAG • Educational Portfolio Project")

if __name__ == "__main__":
    demo.launch()
