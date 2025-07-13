# backend/app.py
from flask import Flask, jsonify, request
from extract_resume import extract_full_resume
from db_handler import save_to_db

app = Flask(__name__)

@app.route("/extract", methods=["POST"])
def extract_and_save():
    # [Optional] Later: Accept file from form
    try:
        extracted_info = extract_full_resume()
        doc_id = save_to_db(extracted_info)
        return jsonify({
            "status": "success",
            "id": doc_id,
            "message": "Data extracted and saved to DB!"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
