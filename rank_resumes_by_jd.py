from pymongo import MongoClient
from utils.comprehensive_ranker import compare_resume_with_jd
from bson import ObjectId

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["resume_ranking_db"]
resumes_collection = db["form_extractions"]
jd_collection = db["jd_extractions"]

def rank_resumes_by_jd():
    """Rank all resumes grouped by JD"""
    
    # Get all JDs
    jds = list(jd_collection.find({}))
    print(f"Found {len(jds)} JDs in database")
    
    if not jds:
        print("No JDs found in database!")
        return
    
    results_by_jd = {}
    
    for jd in jds:
        jd_id = jd["_id"]
        jd_title = jd.get("job_title", "Unknown Position")
        jd_company = jd.get("company_name", "Unknown Company")
        
        print(f"\n{'='*80}")
        print(f"JD: {jd_title}")
        print(f"Company: {jd_company}")
        print(f"JD ID: {jd_id}")
        print(f"{'='*80}")
        
        # Find all resumes that applied for this JD
        resumes_for_jd = list(resumes_collection.find({"jd_id": jd_id}))
        
        if not resumes_for_jd:
            print("No resumes found for this JD")
            continue
        
        print(f"Found {len(resumes_for_jd)} resumes for this JD")
        
        # Rank the resumes for this JD
        ranked_resumes = []
        
        for resume in resumes_for_jd:
            # Compare resume with JD
            comparison_result = compare_resume_with_jd(resume, jd)
            
            ranked_resumes.append({
                "resume_id": str(resume["_id"]),
                "name": resume.get("personal_details", {}).get("name", ""),
                "email": resume.get("personal_details", {}).get("email", ""),
                "phone": resume.get("personal_details", {}).get("phone", ""),
                "score": comparison_result["total_score"],
                "rating": comparison_result["rating"],
                "breakdown": comparison_result["breakdown"]
            })
        
        # Sort by score descending
        ranked_resumes.sort(key=lambda x: x["score"], reverse=True)
        
        # Store results
        results_by_jd[str(jd_id)] = {
            "jd_info": {
                "title": jd_title,
                "company": jd_company,
                "id": str(jd_id)
            },
            "resumes": ranked_resumes
        }
        
        # Display results
        print(f"\nRANKING RESULTS FOR: {jd_title}")
        print(f"{'Rank':<5} {'Score':<8} {'Rating':<15} {'Name':<25} {'Email':<30}")
        print("-" * 85)
        
        for i, result in enumerate(ranked_resumes, 1):
            print(f"{i:<5} {result['score']:<8.1f} {result['rating']:<15} {result['name']:<25} {result['email']:<30}")
        
        # Show detailed breakdown for top 3 candidates
        if ranked_resumes:
            print(f"\nDETAILED BREAKDOWN FOR TOP 3 CANDIDATES:")
            for i, result in enumerate(ranked_resumes[:3], 1):
                print(f"\n{i}. {result['name']} (Score: {result['score']:.1f}/100)")
                breakdown = result['breakdown']
                
                print(f"   Qualification: {breakdown['qualification']['score']}/10 - {breakdown['qualification']['details']}")
                print(f"   Skills: {breakdown['skills']['score']}/25 - {breakdown['skills']['details']}")
                print(f"   Experience: {breakdown['experience']['score']}/20 - {breakdown['experience']['details']}")
                print(f"   Education Quality: {breakdown['education_quality']['score']}/15 - {breakdown['education_quality']['details']}")
                print(f"   Additional: {breakdown['additional']['score']}/10 - {breakdown['additional']['details']}")
    
    return results_by_jd

def show_jd_summary():
    """Show a summary of all JDs and their applicant counts"""
    print(f"\n{'='*80}")
    print("JD SUMMARY")
    print(f"{'='*80}")
    
    jds = list(jd_collection.find({}))
    
    for jd in jds:
        jd_id = jd["_id"]
        jd_title = jd.get("job_title", "Unknown Position")
        jd_company = jd.get("company_name", "Unknown Company")
        
        # Count resumes for this JD
        resume_count = resumes_collection.count_documents({"jd_id": jd_id})
        
        print(f"JD: {jd_title}")
        print(f"Company: {jd_company}")
        print(f"Applicants: {resume_count}")
        print(f"JD Link: /apply/{jd_id}")
        print("-" * 50)

if __name__ == "__main__":
    print("RANKING RESUMES BY JD")
    print("=" * 80)
    
    # Show JD summary first
    show_jd_summary()
    
    # Then rank by JD
    results = rank_resumes_by_jd()
    
    print(f"\n{'='*80}")
    print("RANKING COMPLETED!")
    print(f"{'='*80}") 