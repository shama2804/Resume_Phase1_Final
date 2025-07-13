import os
import re
import fitz
import nltk
import datetime
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import difflib
nltk.download('punkt', quiet=True)
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
GENERIC_SECTIONS = {
    'skills': ['skills', 'technical skills', 'competencies', 'proficiencies', 'tools', 'technologies', 'languages', 'frameworks'],
    'education': ['education', 'academic background', 'qualifications', 'degrees', 'certifications', 'training', 'courses'],
    'projects': ['projects', 'key projects', 'academic projects', 'research projects', 'personal projects'],
    'achievements': ['achievements', 'accomplishments', 'awards', 'publications', 'recognitions', 'certifications',
        'courses', 'training', 'licenses']
}
REPUTED_COLLEGES = set()
with open("reputed_colleges.txt", "r", encoding="utf-8") as f:
    for line in f:
        REPUTED_COLLEGES.add(line.strip().lower())
print("✅ Loaded colleges:", list(REPUTED_COLLEGES)[:5])  # Preview first 5
print(f"✅ Total loaded: {len(REPUTED_COLLEGES)}")

def compute_bonus(candidate, jd_data, relevant_certifications_found=False):
    bonus_factor = 1.0  # Start with no bonus (1.0 multiplier)
    is_fresher_only = jd_data.get("fresher_allowed", False) and not jd_data.get("experienced_allowed", False)
    is_experienced_only = jd_data.get("experienced_allowed", False) and not jd_data.get("fresher_allowed", False)
    is_both_allowed = jd_data.get("fresher_allowed", False) and jd_data.get("experienced_allowed", False)

    experience_list = candidate.get("experience", [])
    name = candidate.get("personal_details", {}).get("name", "Unknown")

    # 🎓 CGPA Bonus (only for freshers)
    try:
        cgpa_str = candidate.get("education", [{}])[0].get("cgpa", "")
        cgpa = float(cgpa_str) if cgpa_str else 0
        if is_fresher_only or is_both_allowed:
            if cgpa >= 9.0:
                bonus_factor *= 1.08  # 8% bonus
                print(f"🎓 CGPA Bonus for {name} → ×1.08")
            elif cgpa >= 8.0:
                bonus_factor *= 1.05  # 5% bonus
                print(f"🎓 CGPA Bonus for {name} → ×1.05")
    except:
        pass

    # 🏅 Certification Bonus
    if relevant_certifications_found:
        if is_fresher_only or is_both_allowed:
            bonus_factor *= 1.08  # 8% bonus
            print(f"🏅 Certification Bonus for {name} → ×1.08")
        elif is_experienced_only:
            bonus_factor *= 1.04  # 4% bonus
            print(f"🏅 Certification Bonus for {name} → ×1.04")

    # 💼 Internship or Experience Bonus
    if is_fresher_only:
        # Only internships matter
        for exp in experience_list:
            role = exp.get("job_title", "").lower()
            company = exp.get("current_company", "").lower()
            if "intern" in role or "intern" in company:
                bonus_factor *= 1.15  # 15% bonus
                print(f"💼 Internship Bonus for {name} (Fresher) → ×1.15")
                break

    elif is_experienced_only:
        # Only full experience matters
        total_years = 0
        for exp in experience_list:
            try:
                from_year = int(exp.get("from_year", 0))
                to_year = int(exp.get("to_year", datetime.datetime.now().year))
                total_years += max(0, to_year - from_year)
            except:
                continue
        if total_years >= 1:
            exp_bonus = min(0.15, 0.03 * total_years)
            bonus_factor *= (1 + exp_bonus)
            print(f"📈 Experience Bonus for {name} → ×{1+exp_bonus:.2f} ({total_years} yrs)")

    elif is_both_allowed:
        # Prefer experience if present; else give internship bonus
        total_years = 0
        for exp in experience_list:
            try:
                from_year = int(exp.get("from_year", 0))
                to_year = int(exp.get("to_year", datetime.datetime.now().year))
                total_years += max(0, to_year - from_year)
            except:
                continue

        if total_years >= 1:
            exp_bonus = min(0.15, 0.03 * total_years)
            bonus_factor *= (1 + exp_bonus)
            print(f"📈 Experience Bonus for {name} (Both allowed) → ×{1+exp_bonus:.2f} ({total_years} yrs)")
        else:
            for exp in experience_list:
                role = exp.get("job_title", "").lower()
                company = exp.get("current_company", "").lower()
                if "intern" in role or "intern" in company:
                    bonus_factor *= 1.15  # 15% bonus
                    print(f"💼 Internship Bonus for {name} (Both allowed) → ×1.15")
                    break

    # Return the multiplicative factor - 1.0 means no bonus
    return bonus_factor - 1.0

def is_reputed_college(college_name, threshold=0.85):
    if not college_name:
        return False
    normalized = college_name.strip().lower()
    
    best_match = None
    best_score = 0

    for rep_college in REPUTED_COLLEGES:
        score = difflib.SequenceMatcher(None, normalized, rep_college).ratio()
        if score > best_score:
            best_score = score
            best_match = rep_college
    if best_score >= threshold:
        return True
    return False  # Silently return False without logging
def compute_certification_relevance(cert_text_block, jd_required_text, jd_nice_text, threshold=0.5):
    if not cert_text_block or not jd_required_text:
        return 0.0

    jd_full = (jd_required_text + " " + jd_nice_text).lower()
    jd_keywords = set(re.findall(r'\b\w{3,}\b', jd_full))

    jd_emb = bi_encoder.encode(jd_full, convert_to_tensor=True)
    cert_lines = [line.strip() for line in cert_text_block.split('\n') if line.strip()]
    total_bonus = 0.0
    relevant_count = 0
    for line in cert_lines:
        cert_emb = bi_encoder.encode(line, convert_to_tensor=True)
        emb_score = util.pytorch_cos_sim(jd_emb, cert_emb).item()

        cert_keywords = set(re.findall(r'\b\w{3,}\b', line.lower()))
        keyword_overlap = len(jd_keywords & cert_keywords) / max(1, len(jd_keywords))

        if emb_score >= threshold or keyword_overlap >= 0.01:
            bonus = min(0.03, 0.015 + 0.02 * emb_score)
            total_bonus += bonus
            relevant_count += 1
    if relevant_count:
        print(f"✅ Found {relevant_count} relevant certifications")
    return min(0.05, total_bonus)
def compute_certification_relevance(cert_text_block, jd_required_text, jd_nice_text, threshold=0.5):
    """
    Scores certifications by relevance to JD using embeddings + keyword overlap.
    """
    if not cert_text_block or not jd_required_text:
        return 0.0

    jd_full = (jd_required_text + " " + jd_nice_text).lower()
    jd_keywords = set(re.findall(r'\b\w{3,}\b', jd_full))

    jd_emb = bi_encoder.encode(jd_full, convert_to_tensor=True)
    cert_lines = [line.strip() for line in cert_text_block.split('\n') if line.strip()]
    total_bonus = 0.0

    for line in cert_lines:
        cert_emb = bi_encoder.encode(line, convert_to_tensor=True)
        emb_score = util.pytorch_cos_sim(jd_emb, cert_emb).item()

        cert_keywords = set(re.findall(r'\b\w{3,}\b', line.lower()))
        keyword_overlap = len(jd_keywords & cert_keywords) / max(1, len(jd_keywords))

        is_relevant = emb_score >= threshold or keyword_overlap >= 0.01  # Either match can trigger

        if is_relevant:
            bonus = min(0.03, 0.015 + 0.02 * emb_score)
            total_bonus += bonus
            print(f"✅ '{line}' → emb_score={emb_score:.2f}, keyword_overlap={keyword_overlap:.2f}, bonus=+{bonus:.2f}")
        else:
            print(f"❌ '{line}' irrelevant → emb_score={emb_score:.2f}, keyword_overlap={keyword_overlap:.2f}")

    return min(0.05, total_bonus)
def compute_education_match(resume_edu_text, jd_qual_text):
    if not resume_edu_text or not jd_qual_text:
        return 0.0
    jd_embedding = bi_encoder.encode(jd_qual_text, convert_to_tensor=True)
    edu_embedding = bi_encoder.encode(resume_edu_text, convert_to_tensor=True)
    return util.pytorch_cos_sim(jd_embedding, edu_embedding).item()
def compute_project_to_role_similarity(projects, jd_summary):
    combined_proj = " ".join(p.get("description", "") for p in projects)
    if not combined_proj or not jd_summary:
        return 0.0
    jd_embedding = bi_encoder.encode(jd_summary, convert_to_tensor=True)
    proj_embedding = bi_encoder.encode(combined_proj.strip(), convert_to_tensor=True)
    return util.pytorch_cos_sim(jd_embedding, proj_embedding).item()
def keyword_presence_bonus(resume_text, keywords):
    if not resume_text or not keywords:
        return 0.0
    hits = sum(1 for word in keywords if word.lower() in resume_text.lower())
    return min(0.03, 0.01 * hits)  

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            blocks = page.get_text("blocks")
            for block in blocks:
                if 0.1 < block[1] / page.rect.height < 0.9:
                    text += block[4] + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
        return ""

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_sections(text, section_definitions):
    text = clean_text(text)
    sections = defaultdict(list)
    current_section = 'other'
    lines = text.split('\n')
    section_patterns = {sec: re.compile(r'\b(' + '|'.join(patterns) + r')\b', re.I) 
                        for sec, patterns in section_definitions.items()}

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        matched = False
        for section, pattern in section_patterns.items():
            if pattern.search(line) and len(line.split()) < 10:
                current_section = section
                matched = True
                continue
                
        sections[current_section].append(line)
    for section in section_definitions:
        if not sections[section] and section != 'other':
            sections[section] = [text]
            
    return {k: ' '.join(v) for k, v in sections.items()}
def calculate_similarity(jd_text, resume_text):
    if not jd_text or not resume_text:
        return 0.0, 0.0, 0.0

    jd_embedding = bi_encoder.encode(jd_text, convert_to_tensor=True)
    resume_embedding = bi_encoder.encode(resume_text, convert_to_tensor=True)
    semantic_score = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()

    jd_words = set(re.findall(r'\b\w{3,}\b', jd_text.lower()))
    resume_words = set(re.findall(r'\b\w{3,}\b', resume_text.lower()))
    overlap = len(jd_words & resume_words)
    keyword_score = min(1.0, overlap / max(1, len(jd_words)))

    combined = (0.75 * semantic_score) + (0.25 * keyword_score)
    scaled = min(0.95, max(0.4, 0.45 + combined * 0.5))

    return scaled, semantic_score, keyword_score

def is_fresher_candidate(form_data):
    try:
        grad_year = int(form_data.get("education", [{}])[0].get("graduation", ""))
        return grad_year >= datetime.datetime.now().year - 2
    except:
        return True


def compute_skill_similarity(candidate_skills, jd_skills_text):
    if not candidate_skills or not jd_skills_text:
        return 0.0
    candidate_str = ' '.join(candidate_skills)
    jd_embedding = bi_encoder.encode(jd_skills_text, convert_to_tensor=True)
    cand_embedding = bi_encoder.encode(candidate_str, convert_to_tensor=True)
    return util.pytorch_cos_sim(jd_embedding, cand_embedding).item()

def compute_project_skill_similarity(projects, jd_skills_text):
    if not projects or not jd_skills_text:
        return 0.0
    combined_text = " ".join(
        p.get("tech_stack", "") + " " + p.get("description", "") for p in projects
    )
    if not combined_text.strip():
        return 0.0
    jd_embedding = bi_encoder.encode(jd_skills_text, convert_to_tensor=True)
    proj_embedding = bi_encoder.encode(combined_text.strip(), convert_to_tensor=True)
    return util.pytorch_cos_sim(jd_embedding, proj_embedding).item()

def compute_section_scores(jd_sections, resume_sections):
    scores = {}
    for section in GENERIC_SECTIONS:
        jd_content = jd_sections.get(section, '')
        resume_content = resume_sections.get(section, '')
        if jd_content and resume_content:
            scores[section], _, _ = calculate_similarity(jd_content, resume_content)
        else:
            scores[section] = 0.0
    return scores
def generate_key_highlights(section_scores, edu_score, project_role_score, cert_bonus, is_reputed):
    highlights = []

    highlights.append(f"Skills Match: {section_scores.get('skills', 0.0):.2f}")
    highlights.append(f"Project Match: {section_scores.get('projects', 0.0):.2f}")
    highlights.append(f"Education Similarity: {edu_score:.2f}")
    highlights.append(f"Project–Role Alignment: {project_role_score:.2f}")

    if cert_bonus > 0:
        highlights.append(f"Certification Bonus: +{cert_bonus:.2f}")
    if is_reputed:
        highlights.append("🎓 Reputed College Bonus: +0.03")

    return highlights


def compute_final_score(section_scores, weights):
    total = 0
    total_weight = 0
    for section, score in section_scores.items():
        weight = weights.get(section, 1.0)
        total += score * weight
        total_weight += weight
    raw_score = total / total_weight if total_weight > 0 else 0
    return min(0.95, max(0.4, 0.45 + raw_score * 0.5))

def analyze_jd(jd_text):
    sections = extract_sections(jd_text, GENERIC_SECTIONS)
    weights = {}
    base_weights = {
        'skills': 2.0,
        'education': 1.5,
        'projects': 1.2,
        'achievements': 1.0
    }
    
    total_words = sum(len(sections.get(s, '').split()) for s in base_weights)
    
    for section, base_weight in base_weights.items():
        content = sections.get(section, '')
        content_words = len(content.split())
        
        weight = base_weight * (0.5 + 0.5 * (content_words / max(total_words, 1)))
        weights[section] = weight
        
        print(f"🔹 JD Section: {section.upper()} - Weight: {weight:.2f}")
    
    return sections, weights

def is_candidate_allowed(candidate, jd_data):
    is_fresher = is_fresher_candidate(candidate["form_data"])
    fresher_ok = jd_data.get("fresher_allowed", False)
    experienced_ok = jd_data.get("experienced_allowed", False)

    if is_fresher and not fresher_ok:
        return False
    if not is_fresher and not experienced_ok:
        return False
    return True

# ranking_logic.py (complete scoring overhaul)

# ranking_logic.py (fixed version)
def compute_deep_score(candidate, jd_data):
    form_data = candidate["form_data"]
    resume_text = candidate["resume_text"]
    resume_sections = candidate["resume_sections"]

    candidate_name = form_data.get("personal_details", {}).get("name", "Unknown")
    projects = form_data.get("projects", [])
    form_skills = form_data.get("skills", [])

    if not is_candidate_allowed(candidate, jd_data):
        return None

    jd_required = jd_data.get("experience_skills", "")
    jd_nice = jd_data.get("nice_to_have_skills", "")
    jd_qualification = jd_data.get("qualification", "")
    jd_summary = jd_data.get("job_summary", "")
    jd_sections = jd_data.get("jd_sections", {})
    jd_weights = jd_data.get("jd_weights", {})

    # --- Core Resume-JD Similarity (most important) ---
    resume_text = clean_text(resume_text)
    jd_full_text = " ".join([
        jd_data.get("job_title", ""),
        jd_data.get("job_summary", ""),
        jd_data.get("responsibilities", ""),
        jd_data.get("experience_skills", ""),
        jd_data.get("nice_to_have_skills", "")
    ])
    jd_full_text = clean_text(jd_full_text)
    
    # Direct similarity between resume and full JD text
    jd_embedding = bi_encoder.encode(jd_full_text, convert_to_tensor=True)
    resume_embedding = bi_encoder.encode(resume_text, convert_to_tensor=True)
    core_similarity = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()
    
    # Start with core similarity as base score
    base_score = core_similarity

    # --- Skill Matching Enhancement ---
    req_match_form = compute_skill_similarity(form_skills, jd_required)
    nice_match_form = compute_skill_similarity(form_skills, jd_nice)
    skill_boost = 0.5 * req_match_form + 0.2 * nice_match_form
    
    # Apply skill boost multiplicatively
    base_score = base_score * (1 + 0.3 * skill_boost)

    # --- Project Relevance ---
    project_relevance = compute_project_to_role_similarity(projects, jd_summary)
    base_score = base_score * (1 + 0.2 * project_relevance)

    # --- Education Match ---
    edu_match = compute_education_match(resume_sections.get("education", ""), jd_qualification)
    base_score = base_score * (1 + 0.1 * edu_match)

    # --- Certification Bonus ---
    cert_bonus = 0.0
    if jd_data.get("certification_appreciated", False):
        resume_cert_text = resume_sections.get("achievements", "")
        cert_bonus = compute_certification_relevance(
            resume_cert_text, jd_required, jd_nice
        )
        base_score = base_score * (1 + 0.5 * cert_bonus)

    # --- Section Similarity ---
    section_score = 0.0
    total_weight = 0.0
    for section in GENERIC_SECTIONS:
        jd_content = jd_sections.get(section, '')
        resume_content = resume_sections.get(section, '')
        if jd_content and resume_content:
            scaled_score, _, _ = calculate_similarity(jd_content, resume_content)
            weight = jd_weights.get(section, 1.0)
            section_score += scaled_score * weight
            total_weight += weight
    
    if total_weight > 0:
        section_score /= total_weight
        base_score = base_score * (1 + 0.2 * section_score)

    # --- Reputed College Bonus ---
    college = form_data.get("education", [{}])[0].get("college", "")
    reputed_bonus = False
    if jd_data.get("filter_by_reputed_colleges", False) and is_reputed_college(college):
        base_score *= 1.05  # 5% bonus
        reputed_bonus = True

    # --- Experience/Fresher Bonus ---
    exp_bonus = compute_bonus(form_data, jd_data, cert_bonus > 0)
    base_score *= (1 + exp_bonus)

    # Final score should be between 0.4-0.97
    final_score = max(0.4, min(0.97, base_score))

    # --- Key Highlights ---
    highlights = [
        f"Core Similarity: {core_similarity:.2f}",
        f"Skill Boost: {skill_boost:.2f}",
        f"Project Relevance: {project_relevance:.2f}",
    ]

    if edu_match > 0:
        highlights.append(f"Education Match: {edu_match:.2f}")
    if section_score > 0:
        highlights.append(f"Section Match: {section_score:.2f}")
    if cert_bonus > 0:
        highlights.append(f"Cert Bonus: {cert_bonus:.2f}")
    if reputed_bonus:
        highlights.append("🎓 Reputed College")
    if exp_bonus > 0:
        highlights.append(f"Exp Bonus: {exp_bonus:.2f}")
    
    return {
        "final_score": round(final_score, 4),
        "name": candidate_name,
        "core_similarity": round(core_similarity, 4),
        "skill_boost": round(skill_boost, 4),
        "project_relevance": round(project_relevance, 4),
        "highlights": highlights
    }
def rank_resumes_for_jd(jd_pdf_path, resumes, jd_data):
    ranked_results = []

    # 📝 Extract JD text and sections
    jd_text = extract_text_from_pdf(jd_pdf_path)
    jd_sections, jd_weights = analyze_jd(jd_text)
    jd_data["jd_sections"] = jd_sections

    # 🚀 Stage 1: Semantic + Keyword similarity
    for resume in resumes:
        name = resume.get("personal_details", {}).get("name", "Unknown")
        print(f"\n📄 Resume: {name}")

        resume_path = resume.get("resume_filepath", "")
        resume_text = extract_text_from_pdf(resume_path)
        resume["resume_text"] = resume_text  # Save for Stage 2

        # ✅ Properly extract resume sections after resume_text is available
        resume_sections = extract_sections(resume_text, GENERIC_SECTIONS)
        resume["resume_sections"] = resume_sections

        # 🎯 Stage 1 similarity scoring
        stage1_scaled, semantic_score, keyword_score = calculate_similarity(jd_text, resume_text)
        resume["semantic_score"] = round(semantic_score, 4)
        resume["keyword_score"] = round(keyword_score, 4)
        resume["stage1_score"] = round(stage1_scaled, 4)

        # 💡 Patch missing form_data
        if "form_data" not in resume:
            resume["form_data"] = {
                "education": resume.get("education", []),
                "experience": resume.get("experience", []),
                "skills": resume.get("skills", []),
                "projects": resume.get("projects", []),
                "links": resume.get("links", {}),
                "personal_details": resume.get("personal_details", {}),
            }

        print(f"🔹 Semantic: {semantic_score:.4f} | Keyword: {keyword_score:.4f} → Stage 1 Score: {stage1_scaled:.4f}")
        ranked_results.append(resume)

    # ✅ Sort Stage 1 results
    ranked_results.sort(key=lambda x: x["stage1_score"], reverse=True)

    # ✅ Print leaderboard
    print("\n🔎 STAGE 1 RESULTS (Semantic + Keyword only):")
    for idx, r in enumerate(ranked_results, 1):
        print(f" {idx}. {r.get('personal_details', {}).get('name', 'Unknown'):25} → Stage 1 Score: {r['stage1_score']:.4f}")

    # ✅ Take top 80% of candidates for Stage 2
    top_80_percent = int(len(ranked_results) * 0.8) or 1
    shortlisted = ranked_results[:top_80_percent]

    # 🧠 Stage 2: Deep Ranking using bonus logic
    final_ranked = []
    for r in shortlisted:
        deep_result = compute_deep_score(r, jd_data)
        if deep_result is not None:
            r["final_score"] = deep_result["final_score"]
            # Remove this line: r["section_scores"] = deep_result["section_scores"]
            r["highlights"] = deep_result["highlights"]
            final_ranked.append(r)

    # ✅ Sort by final deep score
    final_ranked.sort(key=lambda x: x["final_score"], reverse=True)

    # ✅ Print final results
    print("\n🎯 STAGE 2 RESULTS (Final Ranking):")
    for idx, r in enumerate(final_ranked, 1):
        print(f" {idx}. {r.get('personal_details', {}).get('name', 'Unknown'):25} → Final Score: {r['final_score']:.4f}")

    return final_ranked
