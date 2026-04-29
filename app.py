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
        return "❌ Please upload a resume and provide a job description.", "", None, ""
    
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
            messages=[{"role": "user", "content": f"Write a professional cover letter based on this resume and JD:\n\n{resume_text[:1500]}"}]
        )
        cover_letter = cover_res.choices[0].message.content

        # Create download file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(tailored_resume)
            tmp_path = tmp.name

        return analysis, tailored_resume, tmp_path, cover_letter
    except Exception as e:
        return f"❌ Error: {str(e)}", "", None, ""

# --- UI Setup ---
# We use gr.themes.Soft() which is the cleanest professional theme Gradio offers
with gr.Blocks(title="SmartResume AI", theme=gr.themes.Soft(primary_hue="emerald")) as demo:
    
    gr.Markdown("# 📄 SmartResume AI")
    gr.Markdown("### Optimize your resume and stand out to recruiters using RAG-powered AI.")

    with gr.Tabs():
        # Step 1: Input
        with gr.TabItem("Step 1: Upload & Job Details"):
            with gr.Row():
                with gr.Column():
                    resume_input = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
                with gr.Column():
                    job_input = gr.Textbox(label="Job Description", lines=10, placeholder="Paste the full job description here...")
            
            with gr.Row():
                analyze_btn = gr.Button("🚀 Start AI Analysis", variant="primary", size="large")
                example_btn = gr.Button("📝 Load Sample Data", variant="secondary")
                clear_btn = gr.Button("🗑️ Clear All", variant="stop")

        # Step 2: Match Results
        with gr.TabItem("Step 2: Match Analysis"):
            match_output = gr.Textbox(label="Skill Gaps & Suggestions", lines=20, show_copy_button=True)

        # Step 3: Final Documents
        with gr.TabItem("Step 3: Tailored Documents"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### ✨ Tailored Resume")
                    tailored_output = gr.Textbox(label="Optimized Resume Text", lines=15, show_copy_button=True)
                    download_output = gr.File(label="Download Resume (.txt)")
                with gr.Column():
                    gr.Markdown("#### 📧 Generated Cover Letter")
                    cover_letter_output = gr.Textbox(label="Professional Cover Letter", lines=20, show_copy_button=True)

    gr.Markdown("---")
    gr.Markdown("Built with **Groq + RAG** • Educational Portfolio Project")

    # --- Button Logic ---
    def load_sample():
        return None, "Looking for a Senior Python Developer with experience in AI and RAG pipelines."

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
