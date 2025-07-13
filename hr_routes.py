from flask import render_template, request, Blueprint
from bson import ObjectId
from pymongo import MongoClient
import os

from backend.ranking_logic import rank_resumes_for_jd

# Blueprint for ranking routes
hr_bp = Blueprint("ranking_routes", __name__)  # Changed name
# MongoDB collections
client = MongoClient("mongodb://localhost:27017")
jd_collection = client["resume_ranking_db"]["jd_extractions"]
form_collection = client["resume_ranking_db"]["form_extractions"]


@hr_bp.route("/rank_jd/<jd_id>", methods=["GET"])
def rank_jd(jd_id):
    try:
        print(f"📌 Ranking for jd_id: {jd_id}")

        # ✅ Convert jd_id to ObjectId for query
        try:
            jd_object_id = ObjectId(jd_id)
        except Exception:
            return "Invalid JD ID format", 400

        # 🧾 Check for JD PDF
        jd_pdf_path = f"pdfs/JD_{jd_id}_internal.pdf"
        if not os.path.exists(jd_pdf_path):
            return f"❌ JD PDF not found at {jd_pdf_path}", 404

        # 🔍 Load JD from database
        jd_data = jd_collection.find_one({"_id": jd_object_id})
        if not jd_data:
            return "❌ JD not found in database", 404

        # 📥 Load resumes submitted for this JD
        # Change this line
        all_resumes = list(form_collection.find({"jd_id": ObjectId(jd_object_id)}))
        print(f"✅ Matching resumes in form_extractions: {len(all_resumes)}")
        for res in all_resumes:
            print(f"  - Resume ID: {res.get('_id')}  JD_ID: {res.get('jd_id')}")

        if not all_resumes:
            return "⚠️ No resumes submitted for this JD", 200

        # ✅ Rank resumes
        print(f"🏁 Starting ranking for {len(all_resumes)} resumes...")
        final_ranked = rank_resumes_for_jd(jd_pdf_path, all_resumes, jd_data)

        # 🖥️ Render the results on the frontend
        return render_template("ranked_resumes.html", jd_id=jd_id, ranked_resumes=final_ranked)

    except Exception as e:
        print(f"❌ Error during ranking: {str(e)}")
        return f"❌ Internal server error: {str(e)}", 500
