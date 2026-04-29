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

# ================== UI FIX: VISIBLE TEXT FOR WHITE THEME ==================

with gr.Blocks(
    title="SmartResume AI",
    theme=gr.themes.Soft() # Soft theme handles light mode more gracefully
) as demo:
    
    gr.HTML("""
    <style>
        /* Force specific text elements to be dark/black so they are visible on white background */
        
        /* 1. Main Title and Subtitle */
        .prose h1, .prose h2, .prose h3 {
            color: #1a202c !important; /* Dark slate/black */
            text-align: center;
        }
        
        .prose p, .prose strong {
            color: #2d3748 !important; /* Dark grey */
        }

        /* 2. Footer Text */
        .footer-text p {
            color: #4a5568 !important;
            font-weight: 500;
        }

        /* 3. Label Text for inputs */
        label span {
            color: #2d3748 !important;
            font-weight: bold !important;
        }

        /* Keep your button looking cool - orange/green works on white too */
        button.primary {
            background: linear-gradient(90deg, #f97316, #ea580c) !important;
            color: white !important;
            border: none !important;
        }
        
        /* Box borders so they don't vanish on white */
        .block {
            border: 1px solid #e2e8f0 !important;
        }
    </style>
    """)
    
    # Using a container for the header to apply custom classes if needed
    with gr.Column(elem_classes="header-section"):
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

    # Logic functions
    def process_resume(resume_file, job_description):
        analysis, tailored_text, _, cover_letter = analyze_resume(resume_file, job_description)
        if tailored_text and "Error" not in tailored_text:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                tmp.write(f"Tailored Resume\n\n{tailored_text}")
                tmp_path = tmp.name
            return analysis, tailored_text, tmp_path, cover_letter
        return analysis, tailored_text, None, cover_letter

    def load_sample():
        return None, "Looking for a Senior Developer with Python and RAG experience."

    def clear_fields():
        return None, "", "", None, ""

    # Interactivity
    analyze_btn.click(process_resume, [resume_input, job_input], [match_output, tailored_output, download_output, cover_letter_output])
    example_btn.click(load_sample, outputs=[resume_input, job_input])
    clear_btn.click(clear_fields, outputs=[resume_input, job_input, match_output, download_output, cover_letter_output])

    # Footer with custom class for the CSS to target
    with gr.Row(elem_classes="footer-text"):
        gr.Markdown("Built with Groq + RAG • Educational Portfolio Project")

if __name__ == "__main__":
    demo.launch()
