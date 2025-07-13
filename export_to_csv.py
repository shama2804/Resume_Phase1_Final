from pymongo import MongoClient
import pandas as pd

client = MongoClient("mongodb://localhost:27017/")
db = client["resume_ranking_db"]
collection = db["form_extractions"]

# Only get documents not yet exported
data = list(collection.find({"exported": {"$ne": True}}))
print(f" New documents found: {len(data)}")

if not data:
    print(" No new submissions to export.")
else:
    # Clean _id field
    for doc in data:
        doc.pop("_id", None)

    # Normalize and save
    df = pd.json_normalize(data)
    df.to_csv("submissions.csv", mode='a', header=not pd.io.common.file_exists("submissions.csv"), index=False)
    print("New submissions added to CSV.")

    # Mark them as exported
    ids_to_update = [d["doc_id"] for d in data if "doc_id" in d]
    result = collection.update_many({"doc_id": {"$in": ids_to_update}}, {"$set": {"exported": True}})
    print(f" Marked {result.modified_count} as exported.")
