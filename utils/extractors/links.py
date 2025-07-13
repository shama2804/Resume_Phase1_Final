import re
import fitz  # PyMuPDF

def extract_links(text, pdf_path=None):
    linkedin = ""
    website = ""
    social = []

    # 1. Extract visible links from text
    urls = re.findall(r'(https?://[^\s)>\]}]+)', text)
    for url in urls:
        url = url.strip().rstrip('.,)')
        if "linkedin.com" in url:
            linkedin = url
        elif any(domain in url for domain in [
            "github.com", "twitter.com", "instagram.com", "facebook.com",
            "behance.net", "dribbble.com", "medium.com", "youtube.com"
        ]):
            social.append(url)
        else:
            if not website:
                website = url

    # 2. Extract embedded links from PDF (if file path is provided)
    if pdf_path:
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    for link in page.get_links():
                        if "uri" in link:
                            uri = link["uri"]
                            if uri.startswith("http"):
                                if "linkedin.com" in uri and not linkedin:
                                    linkedin = uri
                                elif any(domain in uri for domain in [
                                    "github.com", "portfolio", "notion.so"
                                ]) and not website:
                                    website = uri
                                elif any(s in uri for s in ["twitter", "facebook", "instagram", "youtube"]):
                                    if uri not in social:
                                        social.append(uri)
        except Exception as e:
            print("Error reading embedded links:", e)

    return {
        "linkedin": linkedin,
        "website": website,
        "social": social
    }
