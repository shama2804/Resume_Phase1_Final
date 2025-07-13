import re

def extract_education(text):
    lines = text.split('\n')
    degree_keywords = [
        # Bachelor's Degrees
        "bachelor of arts", "bachelor of science", "bachelor of commerce", "bachelor of business administration", "bachelor of computer applications",
        "bachelor of fine arts", "bachelor of design", "bachelor of architecture", "bachelor of education", "bachelor of engineering", "bachelor of technology",
        "ba", "b.a", "bsc", "b.sc", "b.com", "bcom", "bba", "bbm", "bca", "bfa", "b.des", "b.arch", "b.ed", "b.e", "be", "b.tech", "btech",

        # Master's Degrees
        "master of arts", "master of science", "master of commerce", "master of business administration", "master of computer applications",
        "master of fine arts", "master of design", "master of engineering", "master of technology", "master of philosophy", "master of laws",
        "ma", "m.a", "msc", "m.sc", "m.com", "mcom", "mba", "mib", "mfa", "m.des", "m.e", "me", "m.tech", "mtech", "ms", "m.s", "mphil", "m.phil", "llm", "ll.m",

        # Doctoral Degrees
        "doctor of philosophy", "phd", "ph.d", "dphil", "dsc", "d.litt", "doctorate",

        # Medical Degrees
        "bachelor of medicine, bachelor of surgery", "mbbs", "bachelor of dental surgery", "bds", "bachelor of ayurvedic medicine and surgery", "bams",
        "bachelor of homeopathic medicine and surgery", "bhms", "doctor of medicine", "md", "master of surgery", "ms", "mds",

        # Pharmacy & Allied Health
        "bachelor of pharmacy", "b.pharm", "bpharm", "master of pharmacy", "m.pharm", "mpharm", "bachelor of physiotherapy", "bpt", "master of physiotherapy", "mpt",

        # Law Degrees
        "bachelor of laws", "llb", "b.l", "master of laws", "llm", "ll.m",

        # Education & Teacher Training
        "diploma in education", "d.ed", "bachelor of physical education", "bped", "master of physical education", "mped", "ttc", "b.ed", "m.ed",

        # Management and Postgrad Diplomas
        "pgdm", "post graduate diploma in management", "pgdba", "pgpm", "pgp", "pgpba", "pgdhrm", "mhrm",

        # Architecture & Planning
        "bachelor of planning", "b.plan", "master of planning", "m.plan",

        # Vocational & Certifications
        "diploma", "advanced diploma", "postgraduate diploma", "certificate course", "vocational course",

        # Finance & Accounting
        "chartered accountant", "ca", "icai", "company secretary", "cs", "cfa", "cpa", "acca", "icwa", "cma", "frm", "actuary",

        # Distance / MOOCs
        "nios", "iti", "polytechnic", "ignou", "iim", "iit", "nptel", "coursera", "edx", "udemy", "google certification"
    ]
    for i, line in enumerate(lines):
        clean_line = line.strip().lower()

        for keyword in degree_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', clean_line):
                degree = keyword.upper()

                # Graduation year
                lookahead = ' '.join([
                    lines[i + j].lower() for j in range(1, 4) if i + j < len(lines)
                ])
                year_match = re.search(r'(graduation|completed|passed out|year|batch)?[^0-9]*(20\d{2}|19\d{2})', lookahead)
                graduation_year = year_match.group(2) if year_match else ""

                # CGPA
                search_area = clean_line + " " + lookahead
                match1 = re.search(r'(?:cgpa|gpa)[\s:–-]*([0-9]{1,2}(\.[0-9]+)?)', search_area, re.IGNORECASE)
                match2 = re.search(r'([0-9]{1,2}(\.[0-9]+)?)\s*(?:cgpa|gpa)', search_area, re.IGNORECASE)
                if match1:
                    cgpa = match1.group(1)
                elif match2:
                    cgpa = match2.group(1)
                else:
                    cgpa = ""

                # College extraction
                college = ""
                college_match = re.search(r'(?:from|at)\s+([A-Z][A-Za-z\s,\.\-&()]{5,})', lines[i], re.IGNORECASE)
                if college_match:
                    college = college_match.group(1).strip()
                elif i + 1 < len(lines):
                    potential = lines[i + 1].strip()
                    if 5 < len(potential) < 80 and not any(d in potential.lower() for d in degree_keywords):
                        college = potential.title()

                return [{
                    "degree": degree,
                    "college": college,
                    "graduation": graduation_year,
                    "cgpa": cgpa
                }]
    
    return []