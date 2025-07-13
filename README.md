#  AI-Powered Resume Ranking & Virtual Interview System

This project is an intelligent system that **automatically ranks candidate resumes** for job descriptions (JDs) using **semantic similarity**, **deep section-based analysis**, and **bonus logic** based on CGPA, internships, experience, reputed colleges, and certifications. It also includes a **virtual interview component** with fraud detection (coming soon).

---

##  Features

###  Resume Ranking
- **Stage 1:** Semantic + keyword similarity between JD PDF and Resume PDF using [SBERT](https://www.sbert.net/)
- **Stage 2:** Deep form-based scoring using:
  - Section-wise similarity (skills, projects, education, achievements)
  - CGPA bonuses
  - Certification matching using embeddings
  - Reputed college detection (fuzzy match)
  - Internship/experience-based bonuses

### Bonus Logic
| JD Type | Candidate Experience | Bonus |
|--------|-----------------------|--------|
| Fresher | Internship | `+0.15` |
| Experienced | Years of experience | `+0.03 × years` (capped at `+0.15`) |
| Both Allowed | Internship or Experience | Bonus applied accordingly |

### 📝 JD Form Builder
- HR can submit a JD via a form
- Generates two JD PDFs:
  - Internal (with HR fields)
  - Public (with “Apply Now” link)

### 🧪 Virtual Interview (Phase 2)
> Coming soon!
- Candidate identity verification via **face and voice matching**
- Confidence & communication scoring using Whisper + emotion detection

---

## 🧩 Tech Stack

- **Frontend:** HTML, CSS (Jinja templates via Flask)
- **Backend:** Python, Flask
- **ML/NLP:** Sentence-BERT, PyMuPDF, NLTK
- **Database:** MongoDB
- **Voice/Face Verification (Phase 2):** OpenCV, Whisper

---

## 🗂 Folder Structure
```
PHASE 1-1-GOD/
├── backend/
│ ├── app.py # Resume extraction API
│ ├── db_handler.py # MongoDB insertion logic
│ ├── deep_section_match.py # Section-wise scoring logic
│ ├── extract_resume.py # Resume section extractor
│ └── ranking_logic.py # Full scoring pipeline (Stage 1 + 2)
│
├── pdfs/ # Stores internal & public JD PDFs
│
├── templates/ # HTML templates (Jinja)
│ ├── form.html
│ ├── hr_dashboard.html
│ ├── jd_form.html
│ ├── jd_template_internal.html
│ ├── jd_template_public.html
│ └── ranked_resumes.html
│
├── uploads/ # Resume uploads (PDF)
│
├── utils/
│ └── export_to_csv.py # Download ranked data as CSV
│
├── hr_routes.py # HR dashboard + JD viewing + ranking
├── web_app.py # Resume upload + form flow
├── main.py # Entry point (optional usage)
├── rank_resumes_by_jd.py # Manual CLI-based ranking runner
├── routes.py # Shared route functions (if any)

├── README.md # ← You are here
├── requirements.txt # Dependencies
├── reputed_colleges.txt # List of top-tier colleges
├── submissions.csv # CSV backup of form submissions

```



---

##  Setup Instructions

```bash
# Clone the repo
git clone https://github.com/yourusername/resume-ranking-ai.git
cd resume-ranking-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Create `reputed_colleges.txt` with 100+ top universities
# Start the web app
python web_app.py

 Sample Run
text
Copy code
 STAGE 1 RESULTS (Semantic + Keyword):
 1. John Doe                  → Stage 1 Score: 0.8123
 2. Alice Smith               → Stage 1 Score: 0.7891

 STAGE 2 RESULTS (Final Ranking):
 1. John Doe                  → Final Score: 0.9025
    ↪ Highlights: Skills Match: 0.83, CGPA Bonus +0.03, Internship Bonus +0.15, Certification Bonus +0.05

 2. Alice Smith               → Final Score: 0.8740
    ↪ Highlights: Reputed College Bonus +0.03, Project Match: 0.79
