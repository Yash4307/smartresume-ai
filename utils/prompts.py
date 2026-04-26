def get_analysis_prompt(resume_text, job_description, context):
    return f"""You are an expert career coach and resume optimizer.

Resume Content:
{resume_text[:4000]}

Job Description:
{job_description}

Relevant Context from Resume:
{context}

Provide a clear, professional analysis with:
1. Overall Match Score (0-100%)
2. Key strengths
3. Critical gaps (skills, keywords, experience)
4. Specific, actionable suggestions to improve the resume for this role

Be honest, constructive, and concise."""

def get_tailored_resume_prompt(resume_text, job_description):
    return f"""You are a professional resume writer.

Original Resume:
{resume_text[:3500]}

Target Job Description:
{job_description}

Rewrite and optimize the resume to better match this job.
- Make it ATS-friendly
- Highlight relevant experience and achievements
- Use strong action verbs
- Keep it concise and well-structured

Return only the rewritten resume in clean Markdown format."""