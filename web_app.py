from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import os
from utils.resume_parser import parse_resume
from pymongo import MongoClient
from backend.db_handler import save_to_db
from backend.extract_resume import extract_full_resume
from bson import ObjectId
import pdfkit
import datetime
from routes import hr_bp as jd_blueprint
from hr_routes import hr_bp as ranking_blueprint
flask_app = Flask(__name__)
flask_app.config['UPLOAD_FOLDER'] = 'uploads/'
# ✅ Register both JD and Ranking routes
# Register with prefixes
flask_app.register_blueprint(jd_blueprint, url_prefix='/jd')
flask_app.register_blueprint(ranking_blueprint, url_prefix='/ranking')

client = MongoClient("mongodb://localhost:27017")
db = client["resume_app"]
collection = db["applications"]

def get_default_jd_id():
    """Get the first available JD as default"""
    try:
        jd_client = MongoClient("mongodb://localhost:27017")
        jd_db = jd_client["resume_ranking_db"]
        jd_collection = jd_db["jd_extractions"]
        
        default_jd = jd_collection.find_one({})
        return str(default_jd["_id"]) if default_jd else ""
    except Exception as e:
        print(f"Error getting default JD: {e}")
        return "" 
    
@flask_app.route('/uploads/<filename>')
def send_resume(filename):
    return send_from_directory(flask_app.config['UPLOAD_FOLDER'], filename)

@flask_app.route("/jd_form")
def jd_form_redirect():
    return redirect(url_for("jd_routes.jd_form"))

@flask_app.route('/')
def index():
    default_jd_id = get_default_jd_id()
    return render_template('form.html', prefill={}, resume_filename="", jd_id=default_jd_id)

@flask_app.route('/upload', methods=['POST'])
def upload_resume():
    """Upload and parse resume for auto-fill (DOES NOT SAVE TO DATABASE)"""
    file = request.files['resume']
    jd_id = request.form.get("jd_id", "").strip()
    print(f"Received jd_id from form: '{jd_id}' (length: {len(jd_id)})")
    filename = secure_filename(file.filename) if file.filename else "resume.pdf"
    filepath = os.path.join(flask_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Parse resume for auto-fill
    parsed_data = parse_resume(filepath)
    
    return render_template(
        'form.html',
        prefill=parsed_data,
        resume_filename=filename,
        jd_id=jd_id  # Pass jd_id back to maintain the link
    )

@flask_app.route("/apply/<jd_id>", methods=["GET", "POST"])
def upload_for_jd(jd_id):
    """Direct application for a specific JD"""
    if request.method == 'POST':
        file = request.files['resume']
        filename = secure_filename(file.filename) if file.filename else "resume.pdf"
        filepath = os.path.join(flask_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Parse resume
        parsed_data = parse_resume(filepath)

        # Prepare final data with JD ID
        final_data = {
            **parsed_data,
            "resume_filepath": filepath,
            "jd_id": jd_id,
            "submitted_at": datetime.datetime.utcnow()
        }

        # Save to database
        doc_id = save_to_db(final_data)
        print(f"Saved resume with JD ID: {jd_id}, Doc ID: {doc_id}")

        return jsonify({"message": "Resume submitted successfully", "doc_id": str(doc_id)})

    # GET request - show form with JD ID
    return render_template('form.html', prefill={}, resume_filename="", jd_id=jd_id, url_for=url_for)
@flask_app.route("/submit", methods=["POST"])
def submit():
    """Submit the main application form"""
    # Get form data
    filename = request.form.get("resume_filename")
    jd_id = request.form.get("jd_id", "").strip()
    
    print(f"Received jd_id from form: '{jd_id}'")

    # Validate and get JD ID
    if not jd_id:
        print("WARNING: No jd_id received from form! Using default JD...")
        jd_id = get_default_jd_id()
        if not jd_id:
            return jsonify({"error": "No JD available"}), 400
        print(f"Using default JD ID: {jd_id}")

    # Get filepath if resume was uploaded
    filepath = f"uploads/{filename}" if filename else None

    # Prepare final data
    final_data = {
        "personal_details": {
            "name": request.form.get("name", ""),
            "email": request.form.get("email", ""),
            "phone": request.form.get("phone", ""),
        },
        "education": [{
            "degree": request.form.get("degree", ""),
            "college": request.form.get("college", ""),
            "graduation": request.form.get("graduation", ""),
            "cgpa": request.form.get("cgpa", ""),
        }],
        "experience": [{
            "job_title": request.form.get("experience[0][job_title]", ""),
            "current_company": request.form.get("experience[0][current_company]", ""),
            "employment_duration": request.form.get("experience[0][employment_duration]", ""),
            "job_responsibilities": request.form.get("experience[0][job_responsibilities]", ""),
        }],
        "skills": request.form.getlist("skills[]"),
        "projects": [],
        "links": {
            "linkedin": request.form.get("linkedin", ""),
            "website": request.form.get("website", "")
        },
        "jd_id": jd_id,
        "submitted_at": datetime.datetime.utcnow()
    }

    # Add resume filepath if available
    if filepath:
        final_data["resume_filepath"] = filepath

    # Collect projects dynamically
    i = 0
    while True:
        if f"projects[{i}][title]" not in request.form:
            break
        final_data["projects"].append({
            "title": request.form.get(f"projects[{i}][title]", ""),
            "tech_stack": request.form.get(f"projects[{i}][tech_stack]", ""),
            "description": request.form.get(f"projects[{i}][description]", ""),
            "duration": request.form.get(f"projects[{i}][duration]", ""),
        })
        i += 1
    
    print(f"Final data jd_id: {final_data['jd_id']}")
    
    # Save to database
    doc_id = save_to_db(final_data)
    print(f"Saved application with Doc ID: {doc_id}")

    return jsonify({"message": "Application stored successfully", "doc_id": str(doc_id)})

__all__ = ['flask_app'] 
if __name__ == "__main__":
    flask_app.run(debug=True)
