# utils/eligibility_ranker.py

from bson import ObjectId, errors
import re
from difflib import SequenceMatcher


def normalize_text(text):
    """Normalize text for better comparison"""
    if not text:
        return ""
    return re.sub(r'[^\w\s]', ' ', str(text).lower()).strip()


def calculate_similarity(text1, text2):
    """Calculate similarity between two texts"""
    if not text1 or not text2:
        return 0
    return SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()


def extract_years_from_text(text):
    """Extract years of experience from text"""
    if not text:
        return 0
    
    # Look for patterns like "X years", "X+ years", "X-XX years"
    patterns = [
        r'(\d+)\+?\s*years?',
        r'(\d+)-(\d+)\s*years?',
        r'(\d+)\s*to\s*(\d+)\s*years?'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            if len(matches[0]) == 1:  # Single number
                return int(matches[0][0])
            elif len(matches[0]) == 2:  # Range
                return (int(matches[0][0]) + int(matches[0][1])) / 2
    
    return 0


def check_qualification_match(resume_education, jd_qualification):
    """Check if candidate's qualification matches JD requirements"""
    if not jd_qualification:
        return {"score": 10, "details": "No qualification requirement specified"}
    
    if not resume_education:
        return {"score": 0, "details": "No education information found"}
    
    education_text = ""
    for edu in resume_education:
        degree = edu.get("degree", "")
        college = edu.get("college", "")
        education_text += f"{degree} {college} "
    
    similarity = calculate_similarity(education_text, jd_qualification)
    
    if similarity > 0.7:
        return {"score": 10, "details": f"Qualification matches well (similarity: {similarity:.2f})"}
    elif similarity > 0.4:
        return {"score": 7, "details": f"Qualification partially matches (similarity: {similarity:.2f})"}
    else:
        return {"score": 3, "details": f"Qualification doesn't match well (similarity: {similarity:.2f})"}


def check_skills_match(resume_skills, jd_required_skills, jd_nice_skills):
    """Check skills matching with detailed analysis"""
    if not resume_skills:
        return {"score": 0, "details": "No skills found in resume", "matched_required": [], "matched_nice": []}
    
    resume_skills_set = set(map(str.lower, resume_skills))
    
    # Parse JD skills
    required_skills = []
    nice_skills = []
    
    if jd_required_skills:
        required_skills = [skill.strip().lower() for skill in jd_required_skills.split(",") if skill.strip()]
    if jd_nice_skills:
        nice_skills = [skill.strip().lower() for skill in jd_nice_skills.split(",") if skill.strip()]
    
    # Find exact matches
    matched_required = list(resume_skills_set & set(required_skills))
    matched_nice = list(resume_skills_set & set(nice_skills))
    
    # Find partial matches
    partial_required = []
    partial_nice = []
    
    for skill in resume_skills_set:
        for req_skill in required_skills:
            if req_skill not in matched_required and calculate_similarity(skill, req_skill) > 0.6:
                partial_required.append(f"{skill} ≈ {req_skill}")
        
        for nice_skill in nice_skills:
            if nice_skill not in matched_nice and calculate_similarity(skill, nice_skill) > 0.6:
                partial_nice.append(f"{skill} ≈ {nice_skill}")
    
    # Calculate score
    score = 0
    max_score = len(required_skills) * 3 + len(nice_skills)
    
    if max_score > 0:
        score = (len(matched_required) * 3 + len(matched_nice) + len(partial_required) * 2 + len(partial_nice)) / max_score * 25
        score = min(score, 25)  # Cap at 25 points
    
    details = f"Required skills: {len(matched_required)}/{len(required_skills)} matched, Nice-to-have: {len(matched_nice)}/{len(nice_skills)} matched"
    
    return {
        "score": round(score, 2),
        "details": details,
        "matched_required": matched_required + partial_required,
        "matched_nice": matched_nice + partial_nice,
        "missing_required": [skill for skill in required_skills if skill not in [m.split(" ≈ ")[0] for m in matched_required]]
    }


def check_experience_match(resume_experience, jd_experience_skills):
    """Check experience relevance and duration"""
    if not resume_experience:
        return {"score": 0, "details": "No experience information found"}
    
    total_years = 0
    relevant_experience = []
    
    for exp in resume_experience:
        duration = exp.get("employment_duration", "")
        job_title = exp.get("job_title", "")
        responsibilities = exp.get("job_responsibilities", "")
        
        # Extract years from duration
        years = extract_years_from_text(duration)
        total_years += years
        
        # Check relevance
        experience_text = f"{job_title} {responsibilities}"
        if jd_experience_skills:
            relevance = calculate_similarity(experience_text, jd_experience_skills)
            if relevance > 0.3:
                relevant_experience.append({
                    "title": job_title,
                    "years": years,
                    "relevance": relevance
                })
    
    # Score based on total experience and relevance
    score = 0
    if total_years >= 3:
        score += 10
    elif total_years >= 1:
        score += 7
    elif total_years > 0:
        score += 5
    
    # Bonus for relevant experience
    if relevant_experience:
        max_relevance = max(exp["relevance"] for exp in relevant_experience)
        score += max_relevance * 10
    
    details = f"Total experience: {total_years} years, Relevant roles: {len(relevant_experience)}"
    
    return {"score": round(score, 2), "details": details, "total_years": total_years, "relevant_experience": relevant_experience}


def check_education_quality(resume_education, jd_filter_reputed):
    """Check education quality and college reputation"""
    if not resume_education:
        return {"score": 0, "details": "No education information found"}
    
    score = 0
    details = []
    
    for edu in resume_education:
        college = edu.get("college", "").upper()
        degree = edu.get("degree", "").upper()
        cgpa = edu.get("cgpa", "")
        
        # College reputation check
        reputed_colleges = ["IIT", "NIT", "BITS", "IIIT", "JSSATEB", "MANIPAL", "VIT", "SRM", "AMRITA"]
        is_reputed = any(rep in college for rep in reputed_colleges)
        
        if is_reputed:
            score += 5
            details.append(f"Reputed college: {college}")
        
        # Degree level check
        if "BACHELOR" in degree or "B.E" in degree or "B.TECH" in degree:
            score += 3
            details.append("Bachelor's degree")
        elif "MASTER" in degree or "M.E" in degree or "M.TECH" in degree:
            score += 5
            details.append("Master's degree")
        elif "PHD" in degree or "DOCTORATE" in degree:
            score += 7
            details.append("PhD/Doctorate")
        
        # CGPA check
        if cgpa:
            try:
                cgpa_val = float(cgpa)
                if cgpa_val >= 8.0:
                    score += 3
                    details.append(f"Excellent CGPA: {cgpa}")
                elif cgpa_val >= 7.0:
                    score += 2
                    details.append(f"Good CGPA: {cgpa}")
                elif cgpa_val >= 6.0:
                    score += 1
                    details.append(f"Average CGPA: {cgpa}")
            except:
                pass
    
    return {"score": min(score, 15), "details": "; ".join(details) if details else "Basic education check"}


def check_additional_requirements(resume, jd):
    """Check additional requirements like GitHub, certifications, etc."""
    score = 0
    details = []
    
    # GitHub requirement
    if jd.get("github_required", False):
        github_link = resume.get("links", {}).get("website", "")
        if "github.com" in github_link:
            score += 5
            details.append("GitHub profile found")
        else:
            details.append("GitHub profile required but not found")
    
    # Projects check
    projects = resume.get("projects", [])
    if projects:
        score += 3
        details.append(f"{len(projects)} projects found")
    
    # Certifications (if mentioned in JD)
    if jd.get("show_certification", False):
        # Look for certifications in skills or projects
        skills = resume.get("skills", [])
        projects_text = " ".join([str(p) for p in projects])
        
        cert_keywords = ["certification", "certified", "certificate", "aws", "azure", "gcp", "microsoft", "oracle", "cisco"]
        has_certifications = any(keyword.lower() in " ".join(skills).lower() or keyword.lower() in projects_text.lower() 
                               for keyword in cert_keywords)
        
        if has_certifications:
            score += 3
            details.append("Certifications found")
    
    return {"score": score, "details": "; ".join(details) if details else "No additional requirements"}


def compare_resume_with_jd(resume, jd):
    """Comprehensive comparison of resume with JD"""
    results = {
        "qualification": check_qualification_match(resume.get("education", []), jd.get("qualification")),
        "skills": check_skills_match(resume.get("skills", []), jd.get("experience_skills"), jd.get("nice_to_have_skills")),
        "experience": check_experience_match(resume.get("experience", []), jd.get("experience_skills")),
        "education_quality": check_education_quality(resume.get("education", []), jd.get("filter_by_reputed_colleges")),
        "additional": check_additional_requirements(resume, jd)
    }
    
    # Calculate total score
    total_score = sum(result["score"] for result in results.values())
    
    # Determine overall rating
    if total_score >= 80:
        rating = "Excellent Match"
    elif total_score >= 60:
        rating = "Good Match"
    elif total_score >= 40:
        rating = "Moderate Match"
    elif total_score >= 20:
        rating = "Poor Match"
    else:
        rating = "Not Suitable"
    
    return {
        "total_score": round(total_score, 2),
        "rating": rating,
        "breakdown": results
    }


def rank_resumes(top_resume_ids, resumes_collection, jd_collection):
    """Rank resumes based on comprehensive JD matching"""
    ranked = []

    for resume_id in top_resume_ids:
        # Fetch the resume
        resume = resumes_collection.find_one({"_id": resume_id})
        if not resume:
            print(f"Resume not found for ID: {resume_id}")
            continue

        # Safely convert jd_id to ObjectId
        jd_id = resume.get("jd_id")
        try:
            jd = jd_collection.find_one({"_id": ObjectId(jd_id)})
        except (errors.InvalidId, TypeError):
            print(f"Invalid jd_id for resume {resume_id}: {jd_id}")
            continue

        if not jd:
            print(f"JD not found for ID: {jd_id}")
            continue

        # Comprehensive comparison
        comparison_result = compare_resume_with_jd(resume, jd)
        
        ranked.append({
            "resume_id": str(resume["_id"]),
            "name": resume.get("personal_details", {}).get("name", ""),
            "email": resume.get("personal_details", {}).get("email", ""),
            "phone": resume.get("personal_details", {}).get("phone", ""),
            "score": comparison_result["total_score"],
            "rating": comparison_result["rating"],
            "breakdown": comparison_result["breakdown"]
        })

    # Sort by score descending
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked
