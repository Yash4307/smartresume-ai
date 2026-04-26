import fitz  # PyMuPDF
import io
import os

def extract_text_from_pdf(pdf_input):
    """
    Extract text from PDF.
    Handles:
    1. File path (str) - Default for Gradio
    2. Bytes/Binary data
    3. File-like objects
    """
    try:
        # CASE 1: It's a file path (string) - Most common in Gradio
        if isinstance(pdf_input, str):
            if not os.path.exists(pdf_input):
                return "Error: File path does not exist."
            doc = fitz.open(pdf_input)
            
        # CASE 2: It's raw bytes
        elif isinstance(pdf_input, (bytes, bytearray)):
            doc = fitz.open(stream=pdf_input, filetype="pdf")
            
        # CASE 3: It's a file-like object (has a .read() method)
        elif hasattr(pdf_input, 'name'): # Gradio temp file object
            doc = fitz.open(pdf_input.name)
            
        else:
            return "Error: Unsupported PDF input type."

        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        if not text.strip():
            return "Error: PDF seems empty or is image-based (OCR required)."
            
        return text.strip()

    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_text(text_input):
    """Handle plain text file input"""
    try:
        if isinstance(text_input, str) and os.path.exists(text_input):
            with open(text_input, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return str(text_input).strip()
    except:
        return str(text_input)