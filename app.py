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
        cover_letter_prompt = f"Write a professional, compelling cover letter (280-350 words) based on:\n\nResume:\n{resume_text[:2500]}\n\nJob Description:\n{job_description}"

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

# ================== ULTRA-MODERN CYBER UI ==================
with gr.Blocks(
    title="SmartResume AI",
    theme=gr.themes.Default(primary_hue="cyan", secondary_hue="emerald", neutral_hue="slate")
) as demo:
    
    gr.HTML("""
    <style>
        /* Animated Cyber Background */
        .gradio-container {
            background: linear-gradient(-45deg, #0f172a, #1e1b4b, #020617, #0f172a) !important;
            background-size: 400% 400% !important;
            animation: gradient 15s ease infinite !important;
            min-height: 100vh !important;
            font-family: 'Inter', sans-serif !important;
        }
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Glassmorphism Containers */
        .block, .form {
            background: rgba(30, 41, 59, 0.7) !important;
            backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        }

        /* Visibility & Typography */
        h1 { font-weight: 800 !important; letter-spacing: -1px !important; text-align: center !important; background: linear-gradient(to right, #22d3ee, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .prose p, .prose b { color: #94a3b8 !important; text-align: center !important; }
        
        /* Input & Label Visibility */
        label span { color: #22d3ee !important; font-weight: 600 !important; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 1px; }
        textarea, input { background: rgba(15, 23, 42, 0.6) !important; color: #f8fafc !important; border: 1px solid #334155 !important; border-radius: 8px !important; }
        
        /* Cyber Primary Button */
        .primary-btn {
            background: linear-gradient(90deg, #0891b2, #10b981) !important;
            color: white !important;
            font-weight: 700 !important;
            border: none !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.4) !important;
        }
        .primary-btn:hover { transform: translateY(-2px) !important; box-shadow: 0 0 25px rgba(16, 185, 129, 0.6) !important; }
        
        /* File Upload Styling */
        .file-upload { border: 2px dashed #0891b2 !important; }
    </style>
    """)
    
    with gr.Column(elem_id="header"):
        gr.Markdown("# SmartResume AI")
        gr.Markdown("**Transforming Job Searches with RAG-powered Intelligence**")

    with gr.Row():
        with gr.Column(scale=1):
            resume_input = gr.File(label="📄 1. Upload Resume (PDF)", file_types=[".pdf"], type="binary")
            job_input = gr.Textbox(label="💼 2. Job Description", lines=8, placeholder="Paste the target job description here...")
            analyze_btn = gr.Button("🚀 GENERATE MY TAILORED PACKAGE", variant="primary", elem_classes="primary-btn")

    with gr.Row():
        example_btn = gr.Button("💡 Load Developer Sample", variant="secondary")
        clear_btn = gr.Button("🗑️ Reset Everything", variant="stop")

    with gr.Tabs():
        with gr.TabItem("📊 MATCH ANALYSIS"):
            match_output = gr.Textbox(label="Skill Gap & Keyword Report", lines=13)
        with gr.TabItem("📝 TAILORED RESUME"):
            tailored_output = gr.Textbox(label="Optimized Resume Text", lines=13)
            download_output = gr.File(label="⬇️ Download Document", visible=True)
        with gr.TabItem("📧 COVER LETTER"):
            cover_letter_output = gr.Textbox(label="Professional Application Letter", lines=13)

    # Logic functions
    def process_resume(resume_file, job_description):
        analysis, tailored_text, _, cover_letter = analyze_resume(resume_file, job_description)
        if tailored_text and "Error" not in tailored_text:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
                tmp.write(f"Tailored Resume\n\n{tailored_text}")
                tmp_path = tmp.name
            return analysis, tailored_text, tmp_path, cover_letter
        return analysis, tailored_text, None, cover_letter

    # Button bindings
    analyze_btn.click(process_resume, [resume_input, job_input], [match_output, tailored_output, download_output, cover_letter_output])
    example_btn.click(lambda: (None, "Senior Python Developer role with focus on RAG and LLMs."), outputs=[resume_input, job_input])
    clear_btn.click(lambda: (None, "", "", None, ""), outputs=[resume_input, job_input, match_output, download_output, cover_letter_output])

    gr.Markdown("<center><small>Built with Groq + Llama 3.1 • Powered by RAG Pipeline</small></center>")

if __name__ == "__main__":
    demo.launch()
