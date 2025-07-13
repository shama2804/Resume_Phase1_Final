from backend.db_handler import save_to_db

# Import all your extractors
from personal import extract_personal
from work import extract_work
from education import extract_education
from skills import extract_skills
from links import extract_links
from projects import extract_projects
from experience import extract_experience

# Simulate extracting each part
personal = extract_personal()
work = extract_work()
education = extract_education()
skills = extract_skills()
links = extract_links()
projects = extract_projects()
experience = extract_experience()

# Combine into one final dictionary
extracted_info = {
    **personal,
    **education,
    **work,
    "skills": skills,
    "links": links,
    "projects": projects,
    "experience": experience
}

# Save it to MongoDB
save_to_db(extracted_info)
