import re

def extract_experience(text):
    lines = text.split('\n')
    experience = {
        "total_experience": "",
        "job_title": "",
        "current_company": "",
        "employment_type": "",
        "employment_duration": "",
        "job_responsibilities": "",
        "previous_employers": [],
        "achievements": ""
    }

    # Step 1: Find the 'Experience' section
    start_idx = -1
    for i, line in enumerate(lines):
        if re.search(r'\b(experience|work experience|internship experience|professional experience)\b', line.lower()):
            start_idx = i
            break

    if start_idx == -1:
        return experience  # No experience section found

    # Only process lines under 'Experience' section
    section_lines = lines[start_idx + 1:]

    # Step 2: Extract details
    for i, line in enumerate(section_lines):
        lower = line.lower().strip()

        # Duration
        duration_match = re.search(r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,-]*\d{4})\s*(?:–|to|-)\s*(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,-]*\d{4}|\b(?:present|current))', lower)
        if duration_match:
            experience["employment_duration"] = duration_match.group(0).title()
            continue

        # Job title (broad, for all fields)
        title_match = re.search(r"(intern|assistant|developer|engineer|analyst|consultant|researcher|manager|coordinator|lead|architect)", lower)
        if title_match and not experience["job_title"]:
            experience["job_title"] = title_match.group(1).title()

        # Employment type
        if "intern" in lower:
            experience["employment_type"] = "Internship"
        elif "full-time" in lower or "full time" in lower:
            experience["employment_type"] = "Full-time"
        elif "part-time" in lower or "part time" in lower:
            experience["employment_type"] = "Part-time"

        # Company extraction (fallback)
        if not experience["current_company"]:
            if re.search(r'(at|for)\s+([A-Z][A-Za-z0-9&.\s-]{3,})', line):
                match = re.search(r'(at|for)\s+([A-Z][A-Za-z0-9&.\s-]{3,})', line)
                experience["current_company"] = match.group(2).strip()

        # Responsibilities (capture lines after keywords)
        if "responsibilities" in lower or "key contributions" in lower or "roles" in lower:
            resp_lines = []
            for next_line in section_lines[i + 1 :]:
                if not next_line.strip():
                    break
                resp_lines.append(next_line.strip())
            experience["job_responsibilities"] = " ".join(resp_lines)
            break

    return experience
