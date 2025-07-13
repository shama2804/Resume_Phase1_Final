import re

def extract_personal(text):
    lines = text.split('\n')

    # Email & phone using regex
    email = re.search(r'[\w\.-]+@[\w\.-]+', text)
    phone = re.search(r'\+?\d[\d\s-]{8,15}', text)

    # Heuristic: First line(s) not matching email/phone is usually the name
    name = ""
    for line in lines:
        if email and email.group(0) in line: continue
        if phone and phone.group(0) in line: continue
        if line.strip() and len(line.strip().split()) <= 5:  # avoid summary text
            name = line.strip()
            break

    return {
        "name": name,
        "email": email.group(0) if email else "",
        "phone": phone.group(0) if phone else ""
    }
