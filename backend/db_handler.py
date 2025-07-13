from pymongo import MongoClient
from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["resume_ranking_db"]
collection = db["form_extractions"]

# Function to save extracted form data
def save_to_db(data: dict):
    # Convert jd_id from string to ObjectId if it exists
    jd_id_str = data.get("jd_id", "")
    if jd_id_str:
        try:
            data["jd_id"] = ObjectId(jd_id_str)
            print(f"Converted jd_id to ObjectId: {jd_id_str}")
        except (InvalidId, TypeError):
            print(f"Invalid jd_id format: {jd_id_str}")
            data["jd_id"] = None
    
    data["submitted_at"] = datetime.now(timezone.utc)
    result = collection.insert_one(data)
    print("Saved to DB! ID:", result.inserted_id)
    return str(result.inserted_id)
