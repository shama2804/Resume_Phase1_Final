from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory
from pymongo import MongoClient
from bson import ObjectId
import pdfkit
import os

# Blueprint setup
hr_bp = Blueprint("jd_routes", __name__)  # ✅ Unique name now

# MongoDB setup
client = MongoClient("mongodb://localhost:27017")
db = client["resume_ranking_db"]
jd_collection = db["jd_extractions"]

# wkhtmltopdf configuration
path_wkhtmltopdf = r"D:\needed applications\wkhtmltopdf\bin\wkhtmltopdf.exe"
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# Route: Root test
@hr_bp.route("/hr_test", methods=["GET"])
def root():
   return " HR Blueprint is working!"


# Route: Show JD form
@hr_bp.route("/jd_form", methods=["GET"])
def jd_form():
    print("✅ /jd_form route triggered")  # DEBUG
    return render_template("jd_form.html")


# Route: View JD preview
@hr_bp.route("/view_jd/<jd_id>", methods=["GET"])
def view_jd(jd_id):
    jd_data = jd_collection.find_one({"_id": ObjectId(jd_id)})
    return render_template("jd_template_internal.html", jd=jd_data)

# Route: Submit JD
@hr_bp.route("/submit_jd", methods=["POST"])
def submit_jd():
    # Collect all form data
    data = {
        "job_title": request.form.get("job_title"),
        "employment_type": request.form.get("employment_type"),
        "company_name": request.form.get("company_name"),
        "qualification": request.form.get("qualification"),
        "location": request.form.get("location"),
        "work_mode": request.form.get("work_mode"),
        "about_company": request.form.get("about_company"),
        "job_summary": request.form.get("job_summary"),
        "responsibilities": request.form.get("responsibilities"),
        "experience_skills": request.form.get("experience_skills"),
        "nice_to_have_skills": request.form.get("nice_to_have_skills"),
        "what_to_offer": request.form.get("what_to_offer"),
        "gender": request.form.get("gender"),
        "no_of_candidates": request.form.get("no_of_candidates"),
        "github_required": bool(request.form.get("github_required")),
        "filter_by_reputed_colleges": bool(request.form.get("filter_by_reputed_colleges")),
        "filter_by_reputed_colleges": bool(request.form.get("filter_by_reputed_colleges")),
        "certification_appreciated": "certification_appreciated" in request.form, 
        # ✅ Fresher/Experienced checkboxes
        "fresher_allowed": "fresher_allowed" in request.form,
        "experienced_allowed": "experienced_allowed" in request.form,
    }

    # Optional fields
    if request.form.get("show_reporting_size"):
        data["reporting_size"] = request.form.get("reporting_size")
        data["show_reporting_size"] = True

    if request.form.get("show_stipend"):
        data["stipend"] = request.form.get("stipend")
        data["show_stipend"] = True

    if request.form.get("show_openings"):
        data["no_of_openings"] = request.form.get("no_of_openings")
        data["show_openings"] = True

    if request.form.get("show_certification"):
        data["show_certification"] = True

    # ✅ Save to DB
    inserted = jd_collection.insert_one(data)
    inserted_id = str(inserted.inserted_id)
    data["_id"] = inserted_id

    # Generate PDFs
    internal_html = render_template("jd_template_internal.html", jd=data)
    public_html = render_template("jd_template_public.html", jd=data)

    os.makedirs("pdfs", exist_ok=True)
    internal_path = f"pdfs/JD_{inserted_id}_internal.pdf"
    public_path = f"pdfs/JD_{inserted_id}_public.pdf"

    pdfkit.from_string(internal_html, internal_path, configuration=config)
    pdfkit.from_string(public_html, public_path, configuration=config)
     # Generate URLs with blueprint prefix
    internal_url = url_for("jd_routes_bp.serve_pdf", filename=f"JD_{inserted_id}_internal.pdf")
    public_url = url_for("jd_routes_bp.serve_pdf", filename=f"JD_{inserted_id}_public.pdf")
    
    # In submit_jd route
    return f"""
✅ JD saved successfully!<br><br>
<a href='{url_for('jd_routes.serve_pdf', filename=f"JD_{inserted_id}_internal.pdf")}' target='_blank'>Download Internal JD PDF</a><br>
<a href='{url_for('jd_routes.serve_pdf', filename=f"JD_{inserted_id}_public.pdf")}' target='_blank'>Download Public JD PDF</a><br><br>
<a href='{url_for("jd_routes.view_jd", jd_id=inserted_id)}' target='_blank'>Preview JD</a><br><br>
<a href='{url_for("ranking_routes.rank_jd", jd_id=inserted_id)}' target='_blank'>🔍 View Ranking</a>
"""

# Route: Serve generated PDFs
@hr_bp.route('/pdfs/<filename>')
def serve_pdf(filename):
    return send_from_directory('pdfs', filename)
