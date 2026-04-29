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

        # Match Analysis
        analysis_prompt = get_analysis_prompt(resume_text, job_description, context)
        analysis_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.5,
            max_tokens=800
        )
        analysis = analysis_response.choices[0].message.content

        # Tailored Resume
        tailored_prompt = get_tailored_resume_prompt(resume_text, job_description)
        tailored_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": tailored_prompt}],
            temperature=0.7,
            max_tokens=1200
        )
        tailored_resume = tailored_response.choices[0].message.content

        # Polished Cover Letter
        cover_letter_prompt = f"""Write a professional, compelling cover letter (280-350 words) based on:

Resume:
{resume_text[:2500]}

Job Description:
{job_description}

Requirements:
- Be enthusiastic and confident
- Highlight the most relevant experience and skills
- Make it personalized
- End with a strong call to action"""

        cover_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": cover_letter_prompt}],
            temperature=0.7,
            max_tokens=600
        )
        cover_letter = cover_response.choices[0].message.content

        return analysis, tailored_resume, tailored_resume, cover_letter

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, error_msg, None, None

# ================== DARK THEME CONFIG ==================

# JavaScript to force dark mode on Hugging Face Spaces
js_func = """
function() {
    const url = new URL(window.location);
    url.searchParams.set('__theme', 'dark');
    window.history.replaceState({}, '', url);
}
"""

with gr.Blocks(
    title="SmartResume AI",
    theme=gr.themes.Base(),
    js=js_func
) as demo:
    
    # Custom CSS updated for modern Gradio versions
    gr.HTML("""
    <style>
        /* Force dark background for the entire container */
        .gradio-container {
            background-color: #0a0f1c !important;
            color: #e0f2fe !important;
        }
        
        /* Headers and Text visibility */
        .prose h1, .prose h2, .prose p, .prose strong {
            color: #67e8f9 !important;
        }

        /* Input fields and Cards */
        .block, .form, .gr-box, .border-gray-200, textarea, input {
            background-color: #111827 !important;
            color: #e0f2fe !important;
            border-color: #10b981 !important;
        }

        /* Buttons styling */
        button.primary {
            background: linear-gradient(90deg, #10b981, #059669) !important;
            border: none !important;
            color: white !important;
        }
        
        button.secondary {
            background-color: #1f2937 !important;
            color: #67e8f9 !important;
            border: 1px solid #67e8f9 !important;
        }
        
        /* Fix label text colors */
        label span {
            color: #67e8f9 !important;
            font-weight: bold;
        }
    </style>
    """)
    
    gr.Markdown("# SmartResume AI\n**AI Resume Builder & Job Matcher with RAG**")

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
        else:
            return analysis, tailored_text, None, cover_letter

    def load_sample():
        sample_jd = "We are looking for a Senior Python Developer with strong experience in building web applications using Flask and FastAPI. The ideal candidate should have hands-on experience with LLM integration, RAG pipelines, and prompt engineering."
        return None, sample_jd

    def clear_fields():
        return None, "", "", None, ""

    analyze_btn.click(
        process_resume,
        inputs=[resume_input, job_input],
        outputs=[match_output, tailored_output, download_output, cover_letter_output]
    )

    example_btn.click(
        load_sample,
        inputs=[],
        outputs=[resume_input, job_input]
    )

    clear_btn.click(
        clear_fields,
        inputs=[],
        outputs=[resume_input, job_input, match_output, download_output, cover_letter_output]
    )

    gr.Markdown("---\nBuilt with Groq + RAG • Educational Portfolio Project")

if __name__ == "__main__":
    demo.launch()
