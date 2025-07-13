import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Import all your extractors
from utils.extractors.personal import extract_personal
from utils.extractors.work import extract_work
from utils.extractors.education import extract_education
from utils.extractors.skills import extract_skills
from utils.extractors.links import extract_links
from utils.extractors.projects import extract_projects
from utils.extractors.experience import extract_experience
from utils.resume_parser import parse_resume

# FIXED: This function should only extract data, not save it
def extract_full_resume(filepath):
    """
    Extract all information from resume (DOES NOT SAVE TO DATABASE)
    """
    try:
        # Parse the resume
        parsed_data = parse_resume(filepath)
        
        return {
            "success": True,
            "data": parsed_data
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": {}
        }
