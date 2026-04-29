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

        if "Error" in resume_text:
            return f"❌ {resume_text}", "Could not process resume.", None, None

        context = build_rag_context(resume_text, job_description)

        analysis_prompt = get_analysis_prompt(resume_text, job_description, context)
        analysis_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.5,
            max_tokens=800
        )
        analysis = analysis_response.choices[0].message.content

        tailored_prompt = get_tailored_resume_prompt(resume_text, job_description)
        tailored_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": tailored_prompt}],
            temperature=0.7,
            max_tokens=1200
        )
        tailored_resume = tailored_response.choices[0].message.content

        cover_letter_prompt = f"Write a professional cover letter based on:\nResume: {resume_text[:2000]}\nJD: {job_description}"
        cover_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": cover_letter_prompt}],
            temperature=0.7,
            max_tokens=600
        )
        cover_letter = cover_response.choices[0].message.content
        return analysis, tailored_resume, tailored_resume, cover_letter

    except Exception as e:
        return f"❌ Error: {str(e)}", "Error", None, None

# ================== THE ULTIMATE THEME FIX ==================

with gr.Blocks(
    title="SmartResume AI",
    theme=gr.themes.Soft(primary_hue="orange")
) as demo:
    
    gr.HTML("""
    <style>
        /* This targets the internal variables Gradio 5 uses for text and background */
        :root, .gradio-container {
            --body-background-fill: #0a0f1c !important; /* Deep Dark Blue */
            --block-background-fill: #111827 !important; /* Slightly lighter dark */
            --body-text-color: #e0f2fe !important; /* Off-white text */
            --heading-text-color: #67e8f9 !important; /* Cyan headers */
            --block-label-text-color: #67e8f9 !important; /* Cyan labels */
            --input-background-fill: #1f2937 !important; /* Dark input boxes */
            --border-color-primary: #f97316 !important; /* Orange borders */
        }

        /* Direct overrides for the invisible headers in your screenshot */
        h1, h2, h3, .prose h1, .prose h2, .prose h3 {
            color: #67e8f9 !important;
            text-shadow: 0px 0px 5px rgba(103, 232, 249, 0.3);
        }

        /* Fix the labels like "Upload Resume" and "Job Description" */
        .block span, label span, .block-label {
            color: #67e8f9 !important;
            font-weight: bold !important;
        }

        /* Button Styling */
        button.primary {
            background: linear-gradient(90deg, #f97316, #ea580c) !important;
            color: white !important;
            border: none !important;
        }

        /* Footer visibility */
        .footer-text p {
            color: #94a3b8 !important;
        }
    </style>
    """)
    
    gr.Markdown("# SmartResume AI")
    gr.Markdown("### AI Resume Builder & Job Matcher with RAG")

    with gr.Row():
        with gr.Column(scale=1):
            resume_input = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
            job_input = gr.Textbox(label="Paste Job Description", lines=8, placeholder="Paste the full job description here...")

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
                tmp.write(f"Tailored Resume\n\n{tailored_text}")
                tmp_path = tmp.name
            return analysis, tailored_text, tmp_path, cover_letter
        return analysis, tailored_text, None, cover_letter

    def load_sample():
        return None, "We are looking for a Senior Python Developer with strong experience in building web applications using Flask and FastAPI."

    def clear_fields():
        return None, "", "", None, ""

    analyze_btn.click(process_resume, [resume_input, job_input], [match_output, tailored_output, download_output, cover_letter_output])
    example_btn.click(load_sample, outputs=[resume_input, job_input])
    clear_btn.click(clear_fields, outputs=[resume_input, job_input, match_output, download_output, cover_letter_output])

    with gr.Row(elem_classes="footer-text"):
        gr.Markdown("Built with Groq + RAG • Educational Portfolio Project")

if __name__ == "__main__":
    demo.launch()
