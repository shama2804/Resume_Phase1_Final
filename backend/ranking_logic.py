# ranking_logic.py (optimized and improved)
import os
import re
import fitz
import nltk
import datetime
import time
import pickle
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import difflib
import numpy as np
from functools import lru_cache

# Verbose logging control - set to False for production
VERBOSE = True

nltk.download('punkt', quiet=True)
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Enhanced section definitions with synonyms
GENERIC_SECTIONS = {
    'skills': ['skills', 'technical skills', 'competencies', 'proficiencies', 
               'tools', 'technologies', 'languages', 'frameworks', 'expertise'],
    'education': ['education', 'academic background', 'qualifications', 'degrees', 
                  'certifications', 'training', 'courses', 'academics', 'scholastics'],
    'projects': ['projects', 'key projects', 'academic projects', 'research projects', 
                 'personal projects', 'work experience', 'portfolio', 'initiatives'],
    'achievements': ['achievements', 'accomplishments', 'awards', 'publications', 
                     'recognitions', 'honors', 'certificates', 'licenses', 'credentials']
}

# Reputed colleges - loaded once
REPUTED_COLLEGES = set()
with open("reputed_colleges.txt", "r", encoding="utf-8") as f:
    for line in f:
        REPUTED_COLLEGES.add(line.strip().lower())

if VERBOSE:
    print(f"✅ Loaded {len(REPUTED_COLLEGES)} reputed colleges")

# Cache for JD embeddings
JD_EMBEDDING_CACHE = {}

def log_message(message):
    """Log message if verbose mode is enabled"""
    if VERBOSE:
        print(message)

def compute_bonus(candidate, jd_data, relevant_certifications_found=False):
    """Compute experience/fresher bonus with CGPA normalization"""
    bonus_factor = 1.0
    is_fresher_only = jd_data.get("fresher_allowed", False) and not jd_data.get("experienced_allowed", False)
    is_experienced_only = jd_data.get("experienced_allowed", False) and not jd_data.get("fresher_allowed", False)
    is_both_allowed = jd_data.get("fresher_allowed", False) and jd_data.get("experienced_allowed", False)

    experience_list = candidate.get("experience", [])
    name = candidate.get("personal_details", {}).get("name", "Unknown")

    # 🎓 CGPA Bonus (with normalization)
    try:
        cgpa_str = candidate.get("education", [{}])[0].get("cgpa", "")
        if cgpa_str:
            # Normalize CGPA (convert 4.0 scale to 10.0 scale if needed)
            if '/' in cgpa_str:
                parts = cgpa_str.split('/')
                cgpa = float(parts[0].strip())
                max_scale = float(parts[1].strip())
                if max_scale == 4.0:
                    cgpa = cgpa * 2.5  # Convert to 10-point scale
            else:
                cgpa = float(cgpa_str)
                
            if is_fresher_only or is_both_allowed:
                if cgpa >= 9.0:
                    bonus_factor *= 1.08
                    log_message(f"🎓 CGPA Bonus for {name} → ×1.08")
                elif cgpa >= 8.0:
                    bonus_factor *= 1.05
                    log_message(f"🎓 CGPA Bonus for {name} → ×1.05")
    except Exception as e:
        if VERBOSE:
            print(f"⚠️ CGPA conversion error: {e}")

    # 🏅 Certification Bonus
    if relevant_certifications_found:
        if is_fresher_only or is_both_allowed:
            bonus_factor *= 1.08
            log_message(f"🏅 Certification Bonus for {name} → ×1.08")
        elif is_experienced_only:
            bonus_factor *= 1.04
            log_message(f"🏅 Certification Bonus for {name} → ×1.04")

    # 💼 Experience Bonus (with date standardization)
    if is_fresher_only:
        for exp in experience_list:
            role = exp.get("job_title", "").lower()
            company = exp.get("current_company", "").lower()
            if "intern" in role or "intern" in company:
                bonus_factor *= 1.15
                log_message(f"💼 Internship Bonus for {name} (Fresher) → ×1.15")
                break

    elif is_experienced_only or is_both_allowed:
        total_years = 0
        for exp in experience_list:
            try:
                # Standardize date formats
                duration = exp.get("employment_duration", "")
                if duration:
                    # Handle "MM/YYYY - MM/YYYY" format
                    if '-' in duration:
                        parts = duration.split('-')
                        start = parts[0].strip()
                        end = parts[1].strip()
                        
                        # Parse dates
                        start_date = datetime.datetime.strptime(start, "%m/%Y")
                        end_date = datetime.datetime.strptime(end, "%m/%Y") if end.lower() != "present" \
                            else datetime.datetime.now()
                            
                        total_years += (end_date - start_date).days / 365.25
                    # Handle "YYYY-YYYY" format
                    elif len(duration) == 9 and duration[4] == '-':
                        start_year = int(duration[:4])
                        end_year = int(duration[5:]) if duration[5:] != "Present" \
                            else datetime.datetime.now().year
                        total_years += end_year - start_year
            except Exception as e:
                if VERBOSE:
                    print(f"⚠️ Experience duration error: {e}")
        
        if total_years >= 1:
            exp_bonus = min(0.15, 0.03 * total_years)
            bonus_factor *= (1 + exp_bonus)
            log_message(f"📈 Experience Bonus for {name} → ×{1+exp_bonus:.2f} ({total_years:.1f} yrs)")

    return bonus_factor - 1.0

def is_reputed_college(college_name, threshold=0.85):
    """Check if college is reputed with improved matching"""
    if not college_name:
        return False
    normalized = college_name.strip().lower()
    
    # First check exact match for performance
    if normalized in REPUTED_COLLEGES:
        return True
        
    # Then check for close matches
    best_score = 0
    for rep_college in REPUTED_COLLEGES:
        score = difflib.SequenceMatcher(None, normalized, rep_college).ratio()
        if score > best_score:
            best_score = score
        if best_score >= threshold:
            return True
    return False

def compute_certification_relevance(cert_text_block, jd_required_text, jd_nice_text, threshold=0.5):
    """Improved certification relevance with symmetric keyword matching"""
    if not cert_text_block or not jd_required_text:
        return 0.0

    jd_full = (jd_required_text + " " + jd_nice_text).lower()
    jd_keywords = set(re.findall(r'\b\w{3,}\b', jd_full))
    
    # Precompute JD embedding
    jd_emb = get_jd_embedding(jd_full)
    
    cert_lines = [line.strip() for line in cert_text_block.split('\n') if line.strip()]
    total_bonus = 0.0

    for line in cert_lines:
        cert_emb = bi_encoder.encode(line, convert_to_tensor=True)
        emb_score = util.pytorch_cos_sim(jd_emb, cert_emb).item()

        cert_keywords = set(re.findall(r'\b\w{3,}\b', line.lower()))
        intersection = jd_keywords & cert_keywords
        
        # Symmetric keyword overlap
        if intersection:
            jd_ratio = len(intersection) / max(1, len(jd_keywords))
            cert_ratio = len(intersection) / max(1, len(cert_keywords))
            keyword_overlap = (jd_ratio + cert_ratio) / 2  # Average both ratios
        else:
            keyword_overlap = 0

        is_relevant = emb_score >= threshold or keyword_overlap >= 0.01

        if is_relevant:
            bonus = min(0.03, 0.015 + 0.02 * emb_score)
            total_bonus += bonus
            if VERBOSE:
                print(f"✅ '{line[:50]}' → emb={emb_score:.2f}, kwo={keyword_overlap:.2f}, bonus=+{bonus:.2f}")
        elif VERBOSE:
            print(f"❌ '{line[:50]}' irrelevant → emb={emb_score:.2f}, kwo={keyword_overlap:.2f}")

    return min(0.05, total_bonus)

@lru_cache(maxsize=100)
def get_jd_embedding(jd_text):
    """Get cached JD embedding"""
    return bi_encoder.encode(jd_text, convert_to_tensor=True)

def compute_education_match(resume_edu_text, jd_qual_text):
    if not resume_edu_text or not jd_qual_text:
        return 0.0
    jd_embedding = get_jd_embedding(jd_qual_text)
    edu_embedding = bi_encoder.encode(resume_edu_text, convert_to_tensor=True)
    return util.pytorch_cos_sim(jd_embedding, edu_embedding).item()

def compute_project_to_role_similarity(projects, jd_summary):
    combined_proj = " ".join(p.get("description", "") for p in projects)
    if not combined_proj or not jd_summary:
        return 0.0
        
    # Batch encode for efficiency
    texts_to_encode = [jd_summary, combined_proj.strip()]
    embeddings = bi_encoder.encode(texts_to_encode, convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

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
        log_message(f"❌ Error reading {pdf_path}: {e}")
        return ""

def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_sections(text, section_definitions):
    """Extract resume sections with enhanced synonym handling"""
    text = clean_text(text)
    sections = defaultdict(list)
    current_section = 'other'
    lines = text.split('\n')
    
    # Create inverse mapping from keyword to section
    keyword_to_section = {}
    for section, keywords in section_definitions.items():
        for keyword in keywords:
            keyword_to_section[keyword] = section
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        matched = False
        # Check if line contains any section keyword
        for keyword, section in keyword_to_section.items():
            if re.search(r'\b' + re.escape(keyword) + r'\b', line, re.IGNORECASE):
                if len(line.split()) < 10:  # Avoid long sentences
                    current_section = section
                    matched = True
                    break
                
        sections[current_section].append(line)
    
    # Fallback for missing sections
    for section in section_definitions:
        if not sections[section]:
            sections[section] = [text]
            
    return {k: ' '.join(v) for k, v in sections.items()}

def calculate_similarity(jd_text, resume_text):
    if not jd_text or not resume_text:
        return 0.0, 0.0, 0.0

    # Batch encode for efficiency
    embeddings = bi_encoder.encode([jd_text, resume_text], convert_to_tensor=True)
    semantic_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    jd_words = set(re.findall(r'\b\w{3,}\b', jd_text.lower()))
    resume_words = set(re.findall(r'\b\w{3,}\b', resume_text.lower()))
    overlap = len(jd_words & resume_words)
    keyword_score = min(1.0, overlap / max(1, len(jd_words)))

    combined = (0.75 * semantic_score) + (0.25 * keyword_score)
    scaled = min(0.95, max(0.4, 0.45 + combined * 0.5))

    return scaled, semantic_score, keyword_score

def is_fresher_candidate(form_data):
    """Check if candidate is fresher with date standardization"""
    try:
        grad_date = form_data.get("education", [{}])[0].get("graduation", "")
        if not grad_date:
            return True
            
        # Handle different date formats
        current_year = datetime.datetime.now().year
        
        if len(grad_date) == 4:  # "2023"
            grad_year = int(grad_date)
        elif '-' in grad_date:  # "2020-2024"
            grad_year = int(grad_date.split('-')[-1])
        elif '/' in grad_date:  # "05/2024"
            parts = grad_date.split('/')
            grad_year = int(parts[1]) if len(parts) > 1 else current_year
        else:
            return True
            
        return grad_year >= current_year - 2
    except:
        return True

def compute_skill_similarity(candidate_skills, jd_skills_text):
    if not candidate_skills or not jd_skills_text:
        return 0.0
        
    # Batch encode for efficiency
    texts = [' '.join(candidate_skills), jd_skills_text]
    embeddings = bi_encoder.encode(texts, convert_to_tensor=True)
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

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
        
        if VERBOSE:
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

def compute_deep_score(candidate, jd_data, jd_embeddings_cache):
    """Compute deep score with performance optimizations"""
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

    # --- Core Resume-JD Similarity ---
    jd_full_text = " ".join([
        jd_data.get("job_title", ""),
        jd_data.get("job_summary", ""),
        jd_data.get("responsibilities", ""),
        jd_required,
        jd_nice
    ])
    jd_full_text = clean_text(jd_full_text)
    
    # Use cached embedding if available
    cache_key = hash(jd_full_text)
    if cache_key in jd_embeddings_cache:
        jd_embedding = jd_embeddings_cache[cache_key]
    else:
        jd_embedding = bi_encoder.encode(jd_full_text, convert_to_tensor=True)
        jd_embeddings_cache[cache_key] = jd_embedding
    
    resume_embedding = bi_encoder.encode(resume_text, convert_to_tensor=True)
    core_similarity = util.pytorch_cos_sim(jd_embedding, resume_embedding).item()
    
    base_score = core_similarity

    # --- Skill Matching ---
    req_match_form = compute_skill_similarity(form_skills, jd_required)
    nice_match_form = compute_skill_similarity(form_skills, jd_nice)
    skill_boost = 0.5 * req_match_form + 0.2 * nice_match_form
    base_score *= (1 + 0.3 * skill_boost)

    # --- Project Relevance ---
    project_relevance = compute_project_to_role_similarity(projects, jd_summary)
    base_score *= (1 + 0.2 * project_relevance)

    # --- Education Match ---
    edu_match = compute_education_match(resume_sections.get("education", ""), jd_qualification)
    base_score *= (1 + 0.1 * edu_match)

    # --- Certification Bonus ---
    cert_bonus = 0.0
    if jd_data.get("certification_appreciated", False):
        resume_cert_text = resume_sections.get("achievements", "")
        cert_bonus = compute_certification_relevance(
            resume_cert_text, jd_required, jd_nice
        )
        base_score *= (1 + 0.5 * cert_bonus)

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
        base_score *= (1 + 0.2 * section_score)

    # --- Reputed College Bonus ---
    college = form_data.get("education", [{}])[0].get("college", "")
    reputed_bonus = False
    if jd_data.get("filter_by_reputed_colleges", False) and is_reputed_college(college):
        base_score *= 1.05
        reputed_bonus = True

    # --- Experience/Fresher Bonus ---
    exp_bonus = compute_bonus(form_data, jd_data, cert_bonus > 0)
    base_score *= (1 + exp_bonus)

    # Final score bounds
    final_score = max(0.4, min(0.97, base_score))

    # --- Key Highlights ---
    highlights = [
        f"Core: {core_similarity:.2f}",
        f"Skills: {skill_boost:.2f}",
        f"Projects: {project_relevance:.2f}",
    ]

    if edu_match > 0:
        highlights.append(f"Education: {edu_match:.2f}")
    if section_score > 0:
        highlights.append(f"Sections: {section_score:.2f}")
    if cert_bonus > 0:
        highlights.append(f"Cert: {cert_bonus:.2f}")
    if reputed_bonus:
        highlights.append("🎓 Top College")
    if exp_bonus > 0:
        highlights.append(f"Exp: {exp_bonus:.2f}")
    
    return {
        "final_score": round(final_score, 4),
        "name": candidate_name,
        "highlights": highlights
    }

def rank_resumes_for_jd(jd_pdf_path, resumes, jd_data):
    start_time = time.time()
    ranked_results = []
    jd_embeddings_cache = {}
    
    # Precompute JD text and sections
    jd_text = extract_text_from_pdf(jd_pdf_path)
    jd_sections, jd_weights = analyze_jd(jd_text)
    jd_data["jd_sections"] = jd_sections
    jd_data["jd_weights"] = jd_weights

    # Batch process Stage 1
    batch_texts = []
    for resume in resumes:
        resume_path = resume.get("resume_filepath", "")
        resume_text = extract_text_from_pdf(resume_path)
        resume["resume_text"] = resume_text
        batch_texts.append(resume_text)
        
        # Extract resume sections
        resume_sections = extract_sections(resume_text, GENERIC_SECTIONS)
        resume["resume_sections"] = resume_sections
        
        # Ensure form_data exists
        if "form_data" not in resume:
            resume["form_data"] = {
                "education": resume.get("education", []),
                "experience": resume.get("experience", []),
                "skills": resume.get("skills", []),
                "projects": resume.get("projects", []),
                "links": resume.get("links", {}),
                "personal_details": resume.get("personal_details", {}),
            }

    # Batch compute Stage 1 scores
    jd_emb = bi_encoder.encode(jd_text, convert_to_tensor=True)
    resume_embs = bi_encoder.encode(batch_texts, convert_to_tensor=True)
    
    for i, resume in enumerate(resumes):
        semantic_score = util.pytorch_cos_sim(jd_emb, resume_embs[i]).item()
        
        # Compute keyword score
        jd_words = set(re.findall(r'\b\w{3,}\b', jd_text.lower()))
        resume_words = set(re.findall(r'\b\w{3,}\b', resume["resume_text"].lower()))
        keyword_score = min(1.0, len(jd_words & resume_words) / max(1, len(jd_words)))

        combined = (0.75 * semantic_score) + (0.25 * keyword_score)
        stage1_scaled = min(0.95, max(0.4, 0.45 + combined * 0.5))
        
        resume["semantic_score"] = round(semantic_score, 4)
        resume["keyword_score"] = round(keyword_score, 4)
        resume["stage1_score"] = round(stage1_scaled, 4)
        
        if VERBOSE:
            name = resume.get("personal_details", {}).get("name", "Unknown")
            print(f"📄 {name}: Semantic={semantic_score:.4f}, Keyword={keyword_score:.4f} → Stage1={stage1_scaled:.4f}")

    # Sort Stage 1 results
    ranked_results = sorted(resumes, key=lambda x: x["stage1_score"], reverse=True)
    
    if VERBOSE:
        print("\n🔎 STAGE 1 RESULTS:")
        for idx, r in enumerate(ranked_results, 1):
            name = r.get("personal_details", {}).get("name", "Unknown")
            print(f" {idx}. {name[:20]:20} → Score: {r['stage1_score']:.4f}")

    # Take top candidates for Stage 2
    top_count = max(1, int(len(ranked_results) * 0.8))
    shortlisted = ranked_results[:top_count]
    
    # Process Stage 2 in batches
    batch_size = 8  # Optimal batch size for GPU
    final_ranked = []
    
    for i in range(0, len(shortlisted), batch_size):
        batch = shortlisted[i:i+batch_size]
        for resume in batch:
            deep_result = compute_deep_score(resume, jd_data, jd_embeddings_cache)
            if deep_result is not None:
                resume["final_score"] = deep_result["final_score"]
                resume["highlights"] = deep_result["highlights"]
                final_ranked.append(resume)

    # Sort by final score
    final_ranked.sort(key=lambda x: x["final_score"], reverse=True)
    
    if VERBOSE:
        print("\n🎯 FINAL RANKING:")
        for idx, r in enumerate(final_ranked, 1):
            name = r.get("personal_details", {}).get("name", "Unknown")
            print(f" {idx}. {name[:20]:20} → Score: {r['final_score']:.4f}")
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Processed {len(resumes)} resumes in {elapsed:.2f} seconds")
        print(f"  - Stage 1: {len(resumes)} resumes")
        print(f"  - Stage 2: {len(shortlisted)} resumes")

    return final_ranked
